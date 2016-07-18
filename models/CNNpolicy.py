#coding:utf-8
#import preprocess as ps
#from shogi_ban import GameState

from keras.models import Sequential
from keras.layers import convolutional
from keras.layers.core import Dense, Activation, Flatten
from keras.optimizers import SGD, Adam
import keras.backend as K
from keras.utils import np_utils
from keras.models import model_from_json

import numpy as np

class CNNpolicy(object):
    def __init__(self):
        self.batch_size = 120
        self.nb_classes = 9252 #next hand
        self.nb_epoch = 5
        self.nb_layer = 7

    def make_batch(self):
        # make datasets
        x_dataset, y_dataset = ps.make_sente_datasets(1,100)
        #print(x_dataset[110])
        #print(y_dataset[110])
        x_dataset = np.asarray(x_dataset)
        y_dataset = np.asarray(y_dataset)

        nb_data = x_dataset.shape[0]

        x_train,x_test = np.split(x_dataset,[nb_data*0.9])
        y_train,y_test = np.split(y_dataset,[nb_data*0.9])

        x_train = x_train.reshape(x_train.shape[0], 1, 15, 9)
        x_test = x_test.reshape(x_test.shape[0], 1, 15, 9)

        y_train = np_utils.to_categorical(y_train, nb_classes)
        y_test = np_utils.to_categorical(y_test, nb_classes)
        print("x_train shape:", x_train.shape)
        print(x_train.shape[0], "train samples")
        print(x_test.shape[0], "test samples")

        return x_train, y_train, x_test, y_test

    def create_network(self,):
        network = Sequential()

        # create first layer
        network.add(convolutional.Convolution2D(
                    nb_filter=128,
                    nb_row=5,
                    nb_col=5,
                    input_shape=(1,15,9),
                    init="uniform",
                    activation="relu",
                    border_mode="same"))

        # create all other layers
        for i in range(2, self.nb_layer):
            network.add(convolutional.Convolution2D(
                    nb_filter=128,
                    nb_row=3,
                    nb_col=3,
                    init="uniform",
                    activation="relu",
                    border_mode="same"))

        # the last layer maps each <filters_per_layer> feature to a number
        network.add(convolutional.Convolution2D(
                    nb_filter=1,
                    nb_row=1,
                    nb_col=1,
                    init="uniform",
                    border_mode="same"))

        # reshape output to be board x board
        network.add(Flatten())
        network.add(Dense(128))
        network.add(Activation('relu'))
        #network.add(Dropout(0.5))
        network.add(Dense(self.nb_classes))

        #softmax makes it into a probability distribution
        network.add(Activation("softmax"))

        sgd = SGD(lr=.0003, decay=.0001)
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        network.compile(loss="categorical_crossentropy",
                      optimizer=adam,
                      metrics=["accuracy"])
        return network




if __name__ == "__main__":

    CNNpolicy = CNNpolicy()
    model = CNNpolicy.create_network()
    json_string = model.to_json()

    open('CNNpolicy_architecture.json', 'w').write(json_string)

    print("Can we read model from json file...?")

    model = model_from_json(open('CNNpolicy_architecture.json').read())
    model.load_weights('../parameters/gote_policy_net_weights.hdf5')
    print("...you can!")
