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
import os.path

n2 = hp.puzzleSize
LSTM_output_units = 9
batch_size = 100

train_data = 'data/train_n500000.npz'

def load_data(filename):
    npzfile = np.load(filename)
    return npzfile['quizzes'], npzfile['solutions']

def generate_model():
    # Input layer -> takes in a puzzle matrix
    input_layer = keras.Input(shape=(n2, n2))

    # Concatenate different processing layers
    concat = []
    # Processing layer 1 -> The provided clues
    concat.append(keras.layers.Lambda(lambda x: x)(input_layer))

    # Processing layer 2 -> LSTM on rows
    lstm_row = keras.layers.Bidirectional(keras.layers.LSTM(LSTM_output_units, return_sequences=True))(input_layer)
    concat.append(lstm_row)


    # Processing layer 3 -> LSTM on columns
    transpose = keras.layers.Permute((2, 1))(input_layer)
    lstm_col = keras.layers.Bidirectional(keras.layers.LSTM(LSTM_output_units, return_sequences=True))(transpose)
    concat.append(lstm_col)

    # Processing layer 4 -> LSTM on submatrices
    #transform = keras.layers.Lambda(lambda x: submats_to_rows(x))(input_layer)
    #concat.append(keras.layers.Bidirectional(keras.layers.LSTM(LSTM_output_units, return_sequences=True))(transform))

    # Concatenate each layer
    concat_layer = keras.layers.Concatenate()(concat)
    # TODO: can we avoid hardcoding this?
    concat_reshaped = keras.layers.Reshape((9, 45, 1))(concat_layer);
    conv = keras.layers.Conv2D(hp.num_filters, hp.filter_size, padding='same')(concat_reshaped)
    for i in range(2):
        conv = keras.layers.Conv2D(hp.num_filters, hp.filter_size, padding='same')(conv)
    conv = keras.layers.Conv2D(1, hp.filter_size, padding='same')(conv)
    conv = keras.layers.Reshape((9, 45)) (conv)

    # Dense Layer to convert to n^6 elements
    dense = keras.layers.Dense(n2 ** 2, activation='sigmoid')(conv)
    # Reshape to n^2-by-n^2-by-n^2
    outReshaped=keras.layers.Reshape((n2, n2, n2))(dense)
    # Softmax along last layer
    out = keras.layers.Softmax(axis=-1)(outReshaped)
    
    model = keras.models.Model(inputs=input_layer, outputs=out);
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(model.summary())
    return model

def submats_to_rows(mat):
    print(mat)
    n = hp.cellSize
    rows = np.zeros_like(mat)
    print(rows)
    r = 0
    for i in range(n):
        for j in range(n):
            submat = mat[n*i:n*(i+1), n*j:n*(j+1)]
            print(submat)
            rows[r, :] = np.reshape(submat, (1, n2))
            r = r + 1
    return rows

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

checkpoint_filepath = 'logdir/checkpoints/RNN/'
checkpoint_filename = checkpoint_filepath + 'model.h5'
def training_callbacks():
    if not os.path.exists(checkpoint_filepath):
        os.makedirs(checkpoint_filepath)
    checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_filename, monitor='loss', verbose=1, save_best_only=True)
    return [checkpoint]

def main():
    quizzes, solutions = load_data(train_data)
    solutions = np.array([solution_to_one_hot_tensor(x) for x in solutions])
    train_quizzes, test_quizzes, train_solutions, test_solutions = train_test_split(quizzes, solutions)

    if os.path.isfile(checkpoint_filename):
        print('======LOADING MODEL======')
        model = keras.models.load_model(checkpoint_filename)
    else:
        print('======TRAINING MODEL======')
        model = generate_model()

        model.fit(train_quizzes, train_solutions, batch_size=batch_size, epochs=hp.num_epochs, verbose=1, callbacks = training_callbacks())

    print('Puzzle: \n', test_quizzes[0])
    pred_solutions = model.predict(test_quizzes)
    print('Expected: \n', np.argmax(test_solutions[0], axis=0) + 1)
    s = np.argmax(pred_solutions[0], axis=0) + 1
    print('Actual: \n', s)
   

if __name__ == "__main__":
    main(); print("Done")
