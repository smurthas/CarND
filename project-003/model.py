""" DAVE 2 in Keras """


import sys
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
from PIL import Image
import cv2

# attempt to make the training repeatable. Something is still random, so not
# working yet
random_seed = 1337
np.random.seed(random_seed)
import tensorflow as tf
tf.set_random_seed(random_seed)
sess = tf.Session()

from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.callbacks import ModelCheckpoint


def build_model():
    """ Creates a CNN inspired by NVIDIA's DAVE 2 """
    # Create the Sequential model
    model = Sequential()
    drop_fraction = 0.001

    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(66, 200, 3)))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), input_shape=(66, 200, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(drop_fraction))

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

def process_image(im):
    """ pre-process an image """
    im = np.array(im.resize((200, 160)))
    im = np.reshape(im, (160, 200, 3))
    im = im[64:130, 0:200]
    im = cv2.cvtColor(im, cv2.COLOR_RGB2YUV)
    return im

def image_file_to_array(fn):
    return process_image(Image.open(fn))

def load_data(path, lr=0.25):
    """ loads images and angles from the data set at `path`. Expects
    driving_log.csv to exist in the root dir of `path` and image files to be in
    an IMG subdirectory """

    log_file_name = path + '/driving_log.csv'
    x, angles = [], []

    file = open(log_file_name)
    driving_log = [line.split(',') for line in file][1:]
    file.close()

    image_list = []

    for line in driving_log:
        image_list.append(path + '/IMG/' +line[0].split('/')[-1:][0])
        angle = float(line[3])
        angles.append(angle)
        if lr > 0:
            image_list.append(path + '/IMG/' +line[1].split('/')[-1:][0])
            image_list.append(path + '/IMG/' +line[2].split('/')[-1:][0])
            angles.append(angle+lr)
            angles.append(angle-lr)

    pool = ThreadPool(4)
    x = pool.map(image_file_to_array, image_list)

    return np.reshape(np.array(x), (-1, 66, 200, 3)), angles

def train(training_data_path, model_file, nb_epoch=10):
    """ Creates an instance of the DAVE 2 model and trains it on the X and y
    training sets provided for `nb_epoch` epochs. Saves the .h5 model weights to
    `model_file"""
    model = build_model()
    with open(model_file + '.json', 'w') as model_json:
        model_json.write(model.to_json())
    model.compile('adam', 'mse', ['mean_absolute_error'])

    X_train, y_train = load_data(training_data_path)
    X_train, y_train = shuffle(X_train, y_train, random_state=random_seed)
    checkpoint_callback = ModelCheckpoint(model_file+'_e{epoch:02d}.h5')
    model.fit(
        X_train, y_train,
        batch_size=128, nb_epoch=nb_epoch,
        callbacks=[checkpoint_callback],
        validation_split=0.2
    )

    model.save(model_file + '.h5')

def run():
    """ Load data, train the model, save the weights """
    if len(sys.argv) < 3:
        print("usage: model.py [NB_EPOCHS] [OUTFILE NAME] [INPUT FILES]")
        sys.exit()

    nb_epoch = int(sys.argv[1])
    model_file_name = sys.argv[2]
    training_data_path = sys.argv[3]

    train(training_data_path, model_file_name, nb_epoch=nb_epoch)


if __name__ == '__main__':
    run()
