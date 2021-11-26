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

special_value = -10
LSTM_output_units = 100
n2 = hp.puzzleSize

train_data = 'data/debug_train_3x3_n50.py.npz'

def load_data(filename):
    npzfile = np.load(filename)
    return npzfile['quizzes'], npzfile['solutions']

def generate_model():
    model = keras.Sequential()
    # Not quite sure what this does, just following the in-class exercise
    model.add(keras.layers.Masking(mask_value=special_value, input_shape=(n2, n2)))
    
    # Bidirectional LSTM
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(LSTM_output_units)))

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
    print("sorted copy: ", copy)
    eye = np.eye(len(classes))
    one_hot = np.zeros((n2, n2, n2))

    for i in range(n2):
        for j in range(n2):
            one_hot_encoding = eye[copy[i, j], :]
            one_hot[:, i, j] = one_hot_encoding

    return one_hot


def main():
    model = generate_model()
    quizzes, solutions = load_data(train_data)
    # TODO: test solution_to_one_hot_tensor
    print('quizzes shape: ', len(quizzes))
    # solutions_onehot = [solution_to_one_hot_tensor(x) for x in solutions]
    print('solution:', solutions[0])
    s = solution_to_one_hot_tensor(solutions[0])
    print('solution onehot: ', s)
    print('solution onehot shape: ', s.shape)

if __name__ == "__main__":
    main(); print("Done")
