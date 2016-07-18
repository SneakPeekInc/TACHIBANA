# coding:utf-8
import numpy as np
import os
import time
from copy import deepcopy

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
from TACHIBANA.shogi_ban import GameState

def main():
    cnn_policy = CNNpolicy()
    model = cnn_policy.create_network()
    model = model_from_json(open('../models/CNNpolicy_architecture.json').read())
    sgd = SGD(lr=.03, decay=.0001)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss="categorical_crossentropy",
                  optimizer=adam,
                  metrics=["accuracy"])
    sente_pnn = deepcopy(model)
    gote_pnn = deepcopy(model)
    sente_pnn.load_weights("../parameters/sente_policy_net_weights.hdf5")
    #gote_pnn.load_weights("../parameters/gote_policy_net_weights.hdf5")
    #sente_pnn.load_weights("../parameters/RL_sente_policy_net_weights.hdf5")
    gote_pnn.load_weights("../parameters/RL_gote_policy_net_weights.hdf5")

    while True:
        State = GameState()
        Sente = SenteShogi()
        Gote = GoteShogi()
        states = []
        infos = []
        epsilon = .1

        n = 1
        s_cate = State.sente_category
        g_cate = State.gote_category

        ilegal_state_sente = []
        ilegal_move_sente = []
        ilegal_state_gote = []
        ilegal_move_gote = []

        print(State.board)

        ## is_end_game is True になるまで対局させる ##
        while State.is_end_game is False:
            # 次の手番の判定
            if State.next_player > 0:
                print("info判定")
                # NNで手を選択
                rank = 0
                board = deepcopy(State.board)
                board = board.reshape(1, 1, 15, 9)
                predict = sente_pnn.predict(board)
                argsort = np.argsort(-predict)
                print(argsort[0][0])
                print(argsort[0][1])
                print(argsort[0][2])
                while True:
                    # ε-greedyっぽくしてみる
                    #if (np.random.rand() < epsilon) and (n < 50):
                    #    print("##### !!!!!!!!!! GREEDY !!!!!!!!!!! #####")
                    #    index = argsort[0][rank+1]
                    #else:
                    #    index = argsort[0][rank]
                    index = argsort[0][rank]
                    info = s_cate[index] + [n]
                    rank += 1
                    if rank == cnn_policy.nb_classes:
                        State.is_end_game = True
                        break
                    if info == [(0,0),(0,0),0] + [n]:
                        continue
                    ####### 対戦VERSION #######
                    #info = [(int(input()),int(input())),(int(input()),int(input())),int(input())] + [n]
                    # infoが指せる手であるかの判定
                    if Sente.judge(State.board, info, State.koma_dai_sente):
                        # 局面を一旦currentにコピーしてinfoをもとに局面変更
                        Current = deepcopy(State)
                        Current.update_board(info)
                        #print(Sente.is_ote(D_board))
                        # その手(info)を指したあと王手でなければループを抜ける
                        print(info)
                        if Sente.is_ote(Current.board) is False: break
                    else:
                        tmp = deepcopy(State.board)
                        ilegal_state_sente.append(tmp)
                        info = State.del_turns(info)
                        ilegal_move_sente.append(s_cate.index(info))


            if State.next_player < 0:
                print("info判定")
                # NNで手を選択
                rank = 0
                board = deepcopy(State.board)
                board = board.reshape(1, 1, 15, 9)
                predict = gote_pnn.predict(board)
                argsort = np.argsort(-predict)
                print(argsort[0][0])
                print(argsort[0][1])
                print(argsort[0][2])
                while True:
                    # ε-greedyっぽくしてみる
                    #if (np.random.rand() < epsilon) and (n < 50):
                    #    print("##### !!!!!!!!!! GREEDY !!!!!!!!!!! #####")
                    #    index = argsort[0][rank+1]
                    #else:
                    #    index = argsort[0][rank]
                    index = argsort[0][rank]
                    info = g_cate[index] + [n]
                    rank += 1
                    if rank == cnn_policy.nb_classes:
                        State.is_end_game = True
                        break
                    if info == [(0,0),(0,0),0] + [n]:
                        continue

                    ####### 対戦VERSION #######
                    #info = [(int(input()),int(input())),(int(input()),int(input())),int(input())] + [n]
                    if Gote.judge(State.board, info, State.koma_dai_gote):
                        Current = deepcopy(State)
                        Current.update_board(info)
                        #print(Gote.is_ote(D_board))
                        if Gote.is_ote(Current.board) is False: break

                    else:
                        tmp = deepcopy(State.board)
                        ilegal_state_gote.append(tmp)
                        info = State.del_turns(info)
                        ilegal_move_gote.append(g_cate.index(info))

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
            if n == 255:
                print("とりあえずここまで！")
                State.turn = 0
                break
        print("プラチナむかつく！")

        is_winner = State.turn * (-1)
        ##### RL gote_policy ####
        print(is_winner,"###############  WINNER  ##################")
        ss, si, gs, gi = State.preprocess_for_RL(states,infos)

        if is_winner < 0:
            sente_lr = -1
            gote_lr = 1
            ilegal_state_sente = ilegal_state_sente[0:500]
            ilegal_move_sente = ilegal_move_sente[0:500]

        elif is_winner > 0:
            sente_lr = 1
            gote_lr = -1
            ilegal_state_gote = ilegal_state_gote[0:500]
            ilegal_move_gote = ilegal_move_gote[0:500]

        else:
            sente_lr = -1
            gote_lr = -1
            ss = ss[10:]
            si = si[10:]
            gs = gs[10:]
            gi = gi[10:]
            ilegal_state_sente = ilegal_state_sente[0:100]
            ilegal_move_sente = ilegal_move_sente[0:100]
            ilegal_state_gote = ilegal_state_gote[0:100]
            ilegal_move_gote = ilegal_move_gote[0:100]
        ### for reinforcement learning ##


        ss = np.asarray(ss)
        si = np.asarray(si)
        gs = np.asarray(gs)
        gi = np.asarray(gi)

        ss = ss.reshape(ss.shape[0], 1, 15, 9)
        si = np_utils.to_categorical(si, cnn_policy.nb_classes)
        gs = gs.reshape(gs.shape[0], 1, 15, 9)
        gi = np_utils.to_categorical(gi, cnn_policy.nb_classes)
        #######################################################

        ### for punishing ilegal move ##

        ilss = np.asarray(ilegal_state_sente)
        ilms = np.asarray(ilegal_move_sente)
        ilsg = np.asarray(ilegal_state_gote)
        ilmg = np.asarray(ilegal_move_gote)

        ilss = ilss.reshape(ilss.shape[0], 1, 15, 9)
        ilms = np_utils.to_categorical(ilms, cnn_policy.nb_classes)
        ilsg = ilsg.reshape(ilsg.shape[0], 1, 15, 9)
        ilmg = np_utils.to_categorical(ilmg, cnn_policy.nb_classes)
        ###########################################################


        ############## Reinforcement Learning !!!! #####################

        sente_pnn.optimizer.lr.set_value(sente_lr*0.00001)
        gote_pnn.optimizer.lr.set_value(gote_lr*0.00001)


        sente_pnn.model.fit(ss, si, nb_epoch=1, batch_size=len(ss))
        sente_pnn.save_weights('../parameters/RL_sente_policy_net_weights.hdf5',overwrite=True)

        gote_pnn.model.fit(gs, gi, nb_epoch=1, batch_size=len(gs))
        gote_pnn.save_weights('../parameters/RL_gote_policy_net_weights.hdf5',overwrite=True)


        ###############  Punishing ilegal mobve !!!!  ###################
        print("Punish ilegal move!!")

        cnn_policy = CNNpolicy()
        model = cnn_policy.create_network()
        model = model_from_json(open('../models/CNNpolicy_architecture.json').read())
        sgd = SGD(lr=.03, decay=.0001)
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss="categorical_crossentropy",
                      optimizer=adam,
                      metrics=["accuracy"])
        sente_pnn = deepcopy(model)
        gote_pnn = deepcopy(model)
        sente_pnn.load_weights("../parameters/RL_sente_policy_net_weights.hdf5")
        gote_pnn.load_weights("../parameters/RL_gote_policy_net_weights.hdf5")

        sente_pnn.optimizer.lr.set_value(-0.00001)
        gote_pnn.optimizer.lr.set_value(-0.00001)

        sente_pnn.model.fit(ilss, ilms, nb_epoch=1, verbose=1, batch_size=len(ilss))
        sente_pnn.save_weights('../parameters/RL_sente_policy_net_weights.hdf5',overwrite=True)

        gote_pnn.model.fit(ilsg, ilmg, nb_epoch=1, verbose=1, batch_size=len(ilsg))
        gote_pnn.save_weights('../parameters/RL_gote_policy_net_weights.hdf5',overwrite=True)


if __name__ == "__main__":
    main()
