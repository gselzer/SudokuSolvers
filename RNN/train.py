# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/sudoku
'''
from __future__ import print_function
import tensorflow as tf
from hyperparams import Hyperparams as hp
from data_load import load_data, get_batch_data
from modules import conv
from tqdm import tqdm
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split

special_value = -10
n2 = hp.puzzleSize
LSTM_output_units = 128
batch_size = 100

train_data = 'data/train3x3size100000.npz'

def load_data(filename):
    npzfile = np.load(filename)
    return npzfile['quizzes'], npzfile['solutions']

def generate_model():
    model = keras.Sequential()
    # Not quite sure what this does, just following the in-class exercise
    #model.add(keras.layers.Masking(mask_value=special_value, input_shape=(n2, n2, 1)))

    model.add(keras.layers.Conv2D(hp.num_filters, hp.filter_size, padding='same', activation='relu', input_shape=(n2, n2, 1)))
    
    for i in range(hp.num_blocks - 1):
        model.add(keras.layers.Conv2D(hp.num_filters, hp.filter_size, padding='same', activation='relu'))
    # Bidirectional LSTM
    #model.add(keras.layers.Bidirectional(keras.layers.LSTM(LSTM_output_units)))

    # Dense Layer to convert to n^6 elements
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(n2 ** 3, activation='sigmoid'))
    # Reshape to n^2-by-n^2-by-n^2
    model.add(keras.layers.Reshape((n2, n2, n2)))
    # Softmax along last layer
    model.add(keras.layers.Softmax(axis=-1))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    #print(model.summary())
    return model

# Converts a 2D numpy solution to a 3D tensorflow tensor, where the third dimension is one-hot encoded.
# Modified from https://stackoverflow.com/questions/58406795/how-to-convert-2d-numpy-array-to-one-hot-encoding
def solution_to_one_hot_tensor(solution):
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


def main():
    model = generate_model()
    quizzes, solutions = load_data(train_data)
    quizzes = quizzes.reshape(quizzes.shape[0], n2, n2, 1)
    solutions = np.array([solution_to_one_hot_tensor(x) for x in solutions])
    train_quizzes, test_quizzes, train_solutions, test_solutions = train_test_split(quizzes, solutions)

    model.fit(train_quizzes, train_solutions, batch_size=batch_size, epochs=hp.num_epochs, verbose=1)

    pred_solutions = model.predict(test_quizzes)
    print('Expected: \n', np.argmax(test_solutions[0], axis=0) + 1)
    s = np.argmax(pred_solutions[0], axis=0) + 1
    print('Actual: \n', s)
   

if __name__ == "__main__":
    main(); print("Done")
