# coding:utf-8
import numpy as np
import os
import time
import argparse
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

def make_training_paris(player, mini_batch_size):
    # player: 強化学習させたいほうのplayer。　-1 or 1
    # mini_batch_size: 今のパラメータで行う試合の数

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

    if player == 1:
        sente_pnn.load_weights("./parameters/RL_sente_policy_net_weights.hdf5")
        gote_pnn.load_weights("./parameters/gote_policy_net_weights.hdf5")

    elif player == -1:
        sente_pnn.load_weights("./parameters/sente_policy_net_weights.hdf5")
        gote_pnn.load_weights("./parameters/RL_gote_policy_net_weights.hdf5")

    X_list = []
    y_list = []
    winners = []

    for i in range(mini_batch_size):
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
                board = board.reshape(1, 1, 15, 9)
                predict = sente_pnn.predict(board)
                argsort = np.argsort(-predict)

                while True:
                    index = argsort[0][rank]
                    info = s_cate[index] + [n]
                    rank += 1
                    if rank == cnn_policy.nb_classes:
                        State.is_end_game = True
                        break
                    if info == [(0,0),(0,0),0] + [n]:
                        continue
                    if Sente.judge(State.board, info, State.koma_dai_sente):
                        # 局面を一旦currentにコピーしてinfoをもとに局面変更
                        Current = deepcopy(State)
                        Current.update_board(info)

                        if Sente.is_ote(Current.board) is False: break
                    else:
                        tmp = deepcopy(State.board)
                        info = State.del_turns(info)

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
                    index = argsort[0][rank]
                    info = g_cate[index] + [n]
                    rank += 1
                    if rank == cnn_policy.nb_classes:
                        State.is_end_game = True
                        break
                    if info == [(0,0),(0,0),0] + [n]:
                        continue

                    if Gote.judge(State.board, info, State.koma_dai_gote):
                        Current = deepcopy(State)
                        Current.update_board(info)
                        #print(Gote.is_ote(D_board))
                        if Gote.is_ote(Current.board) is False: break

                    else:
                        tmp = deepcopy(State.board)
                        info = State.del_turns(info)

            state = deepcopy(State.board)
            states.append(state)
            State.update_board(info)

            print(info)
            print(rank)
            print(State.board)
            info = State.del_turns(info)
            infos.append(info)
            n+=1
            if n == 10:
                print("とりあえずここまで！")
                State.turn = 0
                break
        print("プラチナむかつく！")
        is_winner = State.turn * (-1)
        print(is_winner,"###############  WINNER  ##################")

        #winners.append(is_winner)
        winners.append(1)
        X_batch, y_batch = preprocess_for_RL(player,states,infos)
        X_list.append(X_batch)
        y_list.append(y_batch)

    return X_list, y_list, winners


def preprocess_for_RL(player, states, infos):
    path_to_sente_cate = "./preprocessing/sente_category.npy"
    path_to_gote_cate = "./preprocessing/gote_category.npy"
    sente_category = np.load(path_to_sente_cate).tolist()
    gote_category = np.load(path_to_gote_cate).tolist()
    y_batch = list()
    if player == 1:
        X_batch = states[::2]
        infos = infos[::2]
        for i,info in enumerate(infos):
            y_batch.append(sente_category.index(info))
    elif player == -1:
        X_batch = states[1::2]
        infos = infos[1::2]
        for i,info in enumerate(infos):
            y_batch.append(gote_category.index(info))

    X_batch = np.asarray(X_batch).reshape((len(X_batch),1,15,9))
    y_batch = np.asarray(y_batch).reshape((len(y_batch),1))
    y_batch = np_utils.to_categorical(y_batch,9252)

    return X_batch, y_batch


def train_batch(player, X_list, y_list, winners, lr, model):
    for X, y ,winner in zip(X_list, y_list, winners):
        if player == winner:
            model.optimizer.lr.set_value(lr)
        else:
            model.optimizer.lr.set_value(-lr)

        model.train_on_batch(X,y,class_weight=None, sample_weight=None)

def run_training():
    parser = argparse.ArgumentParser(description='Perform reinforcement learning to improve given policy network. Second phase of pipeline.')
    parser.add_argument("--player", "-p", help="Select player who will do RL",type=int)
    parser.add_argument("--learning_rate", "-l", help="Keras learning rate (Default: .03)", type=float, default=.03)
    parser.add_argument("--game_batch", "-g", help="Number of games per mini-batch (Default: 20)", type=int, default=1)
    parser.add_argument("--iterations", "-i", help="Number of training batches/iterations (Default: 10000)", type=int, default=1)
    args = parser.parse_args()
    for i_iter in range(1, args.iterations + 1):
		# Make training pairs and do RL
        X_list, y_list, winners = make_training_paris(args.player, args.game_batch)
        # Set initial conditions
        model = model_from_json(open('./models/CNNpolicy_architecture.json').read())
        sgd = SGD(lr=.03, decay=.0001)
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss="categorical_crossentropy",
                      optimizer=adam,
                      metrics=["accuracy"])
        if args.player == 1:
            path = "./parameters/RL_sente_policy_net_weights.hdf5"
        elif args.player == -1:
            path = "./parameters/RL_gote_policy_net_weights.hdf5"
        model.load_weights(path)
        train_batch(args.player, X_list, y_list, winners, args.learning_rate, model)
		# Save intermediate models
        model.save_weights(path)
if __name__ == "__main__":
    run_training()
