# coding:utf-8
import argparse
import numpy as np
import os
import math
import time
from copy import deepcopy
import sys

from keras.models import Sequential
from keras.layers import convolutional
from keras.layers.core import Dense, Activation, Flatten
from keras.optimizers import SGD, Adam
import keras.backend as K
from keras.utils import np_utils

import json
from keras.models import model_from_json

from TACHIBANA.legal.sente_shogi_legal import SenteShogi
from TACHIBANA.legal.gote_shogi_legal import GoteShogi
from TACHIBANA.models.CNNpolicy import CNNpolicy


SENTE = +1
GOTE  = -1
EMPTY =  0

class GameState(object):
    def __init__(self):
        self.board = self.initial_board()
        self.turn = SENTE
        self.next_player = SENTE
        self.sente = SenteShogi()
        self.gote = GoteShogi()
        self.is_end_game = False
        self.NUM2KANJI_DIC = {
        1:  "歩", -1:  "g歩",
        2:  "香", -2:  "g香",
        3:  "桂", -3:  "g桂",
        4:  "銀", -4:  "g銀",
        5:  "角", -5:  "g角",
        6:  "飛", -6:  "g飛",
        7:  "金", -7:  "g金",
        8:  "玉", -8:  "g玉",
        9:  "と", -9:  "gと",
        10: "杏", -10: "g杏",
        11: "圭", -11: "g圭",
        12: "全", -12: "g全",
        13: "馬", -13: "g馬",
        14: "龍", -14: "g龍",
        0:  "・"}

        #絶対pathでおなしゃす！
        pwd = sys.argv[0]
        path_to_sente_cate = "/home/kosuda/work/TACHIBANA/preprocessing/sente_category.npy"
        path_to_gote_cate = "/home/kosuda/work/TACHIBANA/preprocessing/gote_category.npy"
        self.sente_category = np.load(path_to_sente_cate).tolist()
        self.gote_category = np.load(path_to_gote_cate).tolist()

    @staticmethod
    def initial_board():
        board = np.zeros((11, 9))
        #歩の配置
        board[0:8][6] = 1
        board[0:8][2] = -1
        #香車の配置
        board[8][0],board[8][8] = 2,2
        board[0][0],board[0][8] = -2,-2
        #桂馬の配置
        board[8][1],board[8][7] = 3,3
        board[0][1],board[0][7] = -3,-3
        #銀の配置
        board[8][2],board[8][6] = 4,4
        board[0][2],board[0][6] = -4,-4
        #金の配置
        board[8][3],board[8][5] = 7,7
        board[0][3],board[0][5] = -7,-7
        #王の配置
        board[8][4] = 8
        board[0][4] = -8
        #角の配置
        board[7][1] = 5
        board[1][7] = -5
        #飛車の配置
        board[7][7] = 6
        board[1][1] = -6
        #test
        #board[0][4] = 0
        #board[4][0] = -8
        #board[3][1] = 4
        return board

    def update_board(self,Info,key=1,*board):
        ## 盤上の駒をアップデートする関数 ##
        # Infoは[(before),(after),koma,teban]のリストが入っている。
        # 手番から次に指すプレイヤーを更新
        #Info_convert = lambda Info : Info[0], Info[1], Info[2], Info[3]
        before, after, koma, teban = self.Info_convert(Info)
        self.check_turn(teban)

        # 後手番ならコマにマイナスをかける
        if self.turn > 0:
            koma_dai = self.sente.SENTE_KOMADAI
        elif self.turn < 0:
            koma *= -1
            koma_dai = self.gote.GOTE_KOMADAI

        # "打"に対する挙動
        if before == (-1,9):
            self.board[koma_dai[koma]]-=1 #put_koma()
        else:
            self.board[before] = 0

        # afterの位置に相手駒があった時の操作
        if self.board[after] != 0:
            self.board[koma_dai[self._get_koma(self.board, after)]] += 1

        # 駒を置く
        self.board[after] = koma

        # turnは今のstateを作り出したやつ。
        # next_playerが次指すやつ。
        self.next_player = -1 * self.turn

        # is_end_gameにフラグがたったらgame over
        #if self.is_end_game is True:
            #print("-------GAME OVER!!-------")

    @staticmethod
    def Info_convert(Info):
        return Info[0], Info[1], Info[2], Info[3]


    @staticmethod
    def _get_koma(board, after):
        ## 駒をとった時に呼び出す関数 ##s
        # -1かけて自分の駒にする
        koma = -board[after]
        # もし成駒だったらひっくり返す
        reverse_koma = lambda koma : koma - np.sign(koma)*8 if math.fabs(koma) > 8 else koma

        return reverse_koma(koma)

    def check_turn(self,n):
        if n == 0:
            self.is_end_game = True
        elif n % 2 == 1:
            self.turn = SENTE
        else:
            self.turn = GOTE

    @staticmethod
    def del_turns(ls):
        return ls[:-1]

    def num2kanji(self,board):
        pass

    def preprocess_for_RL(self,states,infos):
        sente_states = states[::2]
        sente_infos = infos[::2]
        gote_states = states[1::2]
        gote_infos = infos[1::2]
        for i,j in enumerate(sente_infos):
            sente_infos[i] = self.sente_category.index(j)
        for i,j in enumerate(gote_infos):
            gote_infos[i] = self.gote_category.index(j)

        return sente_states, sente_infos, gote_states, gote_infos


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='TACHIBANA: Supervised Learinig CNNpolicy')

    parser.add_argument('--battle_mode', '-b', default=0, type=int,
                        help="If you wanna battle, please choose 1 or -1.")

    args = parser.parse_args()

    ### BATTLE MODE ###
    cnn_policy = CNNpolicy()
    model = cnn_policy.create_network()
    model = model_from_json(open('./models/CNNpolicy_architecture.json').read())
    sgd = SGD(lr=.03, decay=.0001)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss="categorical_crossentropy",
                  optimizer=adam,
                  metrics=["accuracy"])
    sente_pnn = deepcopy(model)
    gote_pnn = deepcopy(model)
    sente_pnn.load_weights("./parameters/sente_policy_net_weights.hdf5")
    gote_pnn.load_weights("./parameters/gote_policy_net_weights.hdf5")

    batch_size = 120
    nb_classes = 9252 #next hand
    nb_epoch = 5
    nb_layer = 7


    State = GameState()
    Sente = SenteShogi()
    Gote = GoteShogi()
    states = []
    infos = []

    n = 1
    s_cate = State.sente_category
    g_cate = State.gote_category
    print(State.board)

    ## is_end_game is True になるまで対局させる ##
    while State.is_end_game is False:
        # 次の手番の判定
        if State.next_player > 0:
            print("info判定")
            # NNで手を選択
            rank = 0
            board = deepcopy(State.board)
            board = board.reshape(1, 1, 11, 9)
            predict = sente_pnn.predict(board)
            argsort = np.argsort(-predict)
            print(argsort[0][0])
            print(argsort[0][1])
            print(argsort[0][2])
            while True:
                index = argsort[0][rank]
                info = s_cate[index] + [n]
                rank += 1
                if rank == cnn_policy.nb_classes:
                    State.is_end_game = True
                    break
                if info == [(0,0),(0,0),0] + [n]:
                    continue

                ####### 対戦VERSION #######
                if args.battle_mode == 1:
                    while True:
                        try:
                            info = [(int(input()),int(input())),(int(input()),int(input())),int(input())] + [n]
                            if info == [(0,0),(0,0),0] + [n]:
                                print("投了します")
                                sys.exit()
                            break
                        except ValueError:
                            print("もう一回！")

                # infoが指せる手であるかの判定
                if Sente.judge(State.board, info):
                    # 局面を一旦currentにコピーしてinfoをもとに局面変更
                    Current = deepcopy(State)
                    Current.update_board(info)
                    #print(Sente.is_ote(D_board))
                    # その手(info)を指したあと王手でなければループを抜ける
                    print(info)
                    if Sente.is_ote(Current.board) is False: break

        if State.next_player < 0:
            print("info判定")
            # NNで手を選択
            rank = 0
            board = deepcopy(State.board)
            board = board.reshape(1, 1, 11, 9)
            predict = gote_pnn.predict(board)
            argsort = np.argsort(-predict)
            print(argsort[0][0])
            print(argsort[0][1])
            print(argsort[0][2])
            while True:
                index = argsort[0][rank]
                info = g_cate[index] + [n]
                rank += 1
                if rank == cnn_policy.nb_classes:
                    State.is_end_game = True
                    break
                if info == [(0,0),(0,0),0] + [n]:
                    continue

                ####### 対戦VERSION #######

                if args.battle_mode == -1:
                    while True:
                        try:
                            info = [(int(input()),int(input())),(int(input()),int(input())),int(input())] + [n]
                            if info == [(0,0),(0,0),0] + [n]:
                                print("投了します")
                                sys.exit()
                            break
                        except ValueError:
                            print("もう一回！")

                if Gote.judge(State.board, info):
                    #print(State.koma_dai_gote)
                    Current = deepcopy(State)
                    Current.update_board(info)
                    #print(Gote.is_ote(D_board))
                    if Gote.is_ote(Current.board) is False: break
        state = deepcopy(State.board)
        states.append(state)
        State.update_board(info)
        print("詰み判定")
        # ボードのみをコピーして詰みのジャッジをする(Stateに干渉しないボード)

        print("判定終了")
        print(info)
        print(rank)
        print(State.board)
        info = State.del_turns(info)
        #print(info)
        infos.append(info)

        #time.sleep(1)
        n+=1
        if n == 400:
            print("とりあえずここまで！")
            State.turn = 0
            break
    #print(infos)
    is_winner = State.turn * (-1)
    print("## WINNER {} ##".format(is_winner))
    #print("プラチナむかつく！")
