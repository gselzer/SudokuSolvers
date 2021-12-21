# -*- coding: utf-8 -*-
#/usr/bin/python2

"""
3-layer CNN adapted from 
https://towardsdatascience.com/solving-sudoku-with-convolution-neural-network-keras-655ba4be3b11
Author: Gabriel Selzer
"""

import tensorflow as tf
from tensorflow import keras
from hyperparams import Hyperparams as hp
import numpy as np
import os.path
import tensorflow.keras.backend as K

n2 = hp.puzzleSize
LSTM_output_units = 9
batch_size = hp.batch_size

#train_data = 'data/debug_n100.npz'
train_data = '../CNN/data/train500ksize3.npz'

def load_data(filename):
    npzfile = np.load(filename)
    return npzfile['quizzes'], npzfile['solutions']

def generate_model():
    # Input layer -> takes in a puzzle matrix
    input_layer = keras.Input(shape=(n2, n2))
    input_reshape = keras.layers.Reshape((n2, n2, 1))(input_layer)
    input_norm = keras.layers.BatchNormalization()(input_reshape)

    conv1 = keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(input_norm)
    norm1 = keras.layers.BatchNormalization()(conv1)

    conv2 = keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(norm1)
    norm2 = keras.layers.BatchNormalization()(conv2)

    conv3 = keras.layers.Conv2D(128, kernel_size=(1,1), activation='relu', padding='same')(norm2)
    norm3 = keras.layers.BatchNormalization()(conv3)

    flat = keras.layers.Flatten()(norm3)
    dense = keras.layers.Dense(n2 ** 3)(flat)

    out_reshape = keras.layers.Reshape((-1, n2))(dense)
    act = keras.layers.Activation('softmax')(out_reshape)

    model = keras.models.Model(inputs=input_layer, outputs=act);
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    print(model.summary())
    return model

# Converts a 2D numpy solution to a 3D tensorflow tensor, where the third dimension is one-hot encoded.
# Modified from https://stackoverflow.com/questions/58406795/how-to-convert-2d-numpy-array-to-one-hot-encoding
def one_hot(solution):
    classes = range(1, n2 + 1)
    copy = np.copy(solution)
    copy = np.searchsorted(classes, copy)
    eye = np.eye(len(classes))
    one_hot = np.zeros((n2, n2, n2))

    for i in range(n2):
        for j in range(n2):
            one_hot_encoding = eye[copy[i, j], :]
            one_hot[:, i, j] = one_hot_encoding

    return one_hot.reshape(n2, n2, n2)

def arg_max(solution):
    return np.argmax(solution, axis=1) + 1

checkpoint_filepath = './RNN/'
checkpoint_filename = checkpoint_filepath + 'model.h5'
def training_callbacks():
    if not os.path.exists(checkpoint_filepath):
        os.makedirs(checkpoint_filepath)
    checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_filename, monitor='loss', verbose=1, save_best_only=True)
    return [checkpoint]

def main():
    quizzes, solutions = load_data(train_data)
    solutions = solutions - 1

    if os.path.isfile(checkpoint_filename):
        print('======LOADING MODEL======')
        model = keras.models.load_model(checkpoint_filename)
    else:
        print('======TRAINING MODEL======')
        model = generate_model()

        model.fit(quizzes, solutions, batch_size=batch_size, epochs=hp.num_epochs, verbose=1, callbacks = training_callbacks())
   

if __name__ == "__main__":
    main(); print("Done")
