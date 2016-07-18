import argparse
import time

from keras.models import Sequential
from keras.layers import convolutional
from keras.layers.core import Dense, Activation, Flatten
from keras.optimizers import SGD, Adam
import keras.backend as K
from keras.utils import np_utils
from keras.models import model_from_json

import numpy as np
import json

from TACHIBANA.shogi_ban import GameState
import TACHIBANA.preprocessing.preprocess as ps
from TACHIBANA.models.CNNpolicy import CNNpolicy




parser = argparse.ArgumentParser(description='TACHIBANA: Supervised Learinig CNNpolicy')

parser.add_argument('--player', '-p', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')

args = parser.parse_args()

cnn_policy = CNNpolicy()
model = cnn_policy.create_network()
model = model_from_json(open('../models/CNNpolicy_architecture.json').read())
sgd = SGD(lr=.03, decay=.0001)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss="categorical_crossentropy",
              optimizer=adam,
              metrics=["accuracy"])

error = "###### You have to choose player you wanna train!! ######"
assert args.player==1 or args.player==-1, error

if args.player == 1:
    model.load_weights("../parameters/sente_policy_net_weights.hdf5")
    path = "../parameters/sente_policy_net_weights.hdf5"
    x_dataset, y_dataset = ps.make_sente_datasets(1,3)

elif args.player == -1:
    model.load_weights("../parameters/gote_policy_net_weights.hdf5")
    path = "../parameters/gote_policy_net_weights.hdf5"
    x_dataset, y_dataset = ps.make_gote_datasets(1,3)

x_dataset = np.asarray(x_dataset)
y_dataset = np.asarray(y_dataset)

nb_data = x_dataset.shape[0]

x_train,x_test = np.split(x_dataset,[nb_data*0.9])
y_train,y_test = np.split(y_dataset,[nb_data*0.9])

x_train = x_train.reshape(x_train.shape[0], 1, 15, 9)
x_test = x_test.reshape(x_test.shape[0], 1, 15, 9)

y_train = np_utils.to_categorical(y_train, cnn_policy.nb_classes)
y_test = np_utils.to_categorical(y_test, cnn_policy.nb_classes)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

## train
model.fit(x_train,y_train,
            batch_size=cnn_policy.batch_size,
            nb_epoch=cnn_policy.nb_epoch,
            verbose=1,
            validation_data=(x_test, y_test))

model.save_weights(path)

## check result of training
state = GameState()

## print move and probability ranking ##
board = state.board.reshape(1, 1, 15, 9)
check = model.predict(board)

argsort = np.argsort(-check)
check[0][:] = check[0][argsort]

print("1st",argsort[0][0])
print("probability:",check[0][0])
print("2nd",argsort[0][1])
print("probability:",check[0][1])
print("3rd",argsort[0][2])
print("probability:",check[0][2])
