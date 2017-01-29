""" DAVE 2 in Keras """

import sys

import pickle
import numpy as np

from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D


def build_model():
    # Create the Sequential model
    model = Sequential()

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), input_shape=(66, 200, 3)))
    model.add(Activation('relu'))

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
    model.add(Activation('relu'))

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))

    model.add(Dense(1))

    return model


def train(X_train, y_train, model_file, nb_epoch=10):
    model = build_model()
    model.compile('adam', 'mse', ['mean_absolute_error'])
    model.fit(X_train, y_train, batch_size=128, nb_epoch=nb_epoch, validation_split=0.2)
    model.save(model_file)

def load_training_datas(filenames):
    X_train = []
    y_train = []
    for filename in filenames:
        with open(filename, 'rb') as f:
            print('Loading', filename, '...')
            tmp_data = pickle.load(f)
            X_train.extend(tmp_data['features'])
            y_train.extend(tmp_data['angles'])
    return shuffle(X_train, y_train)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: sms_model.py [NB_EPOCHS] [OUTFILE NAME] [INPUT FILES]")
        sys.exit()

    nb_epoch = int(sys.argv[1])
    model_file_name = sys.argv[2]
    data_files = sys.argv[3:]

    X_train, y_train = load_training_datas(data_files)
    X_train = np.reshape(np.array(X_train), (-1, 66, 200, 3))
    train(X_train, y_train, model_file_name, nb_epoch=nb_epoch)

