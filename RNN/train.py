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
import tensorflow.keras.backend as K

n2 = hp.puzzleSize
LSTM_output_units = 9
batch_size = 1000

train_data = 'data/train_n500000.npz'

def load_data(filename):
    npzfile = np.load(filename)
    return npzfile['quizzes'], npzfile['solutions']

def submatrices(mat):
    r, h = mat.shape
    return (mat.reshape(h//3, 3, -1, 3)
            .swapaxes(1,2)
            .reshape(-1, 3, 3))

def input_loss(puzzle, is_training):
    
    def loss(y_true, y_pred):
        blank_indices = np.where(puzzle == 0)
        pred = tf.math.add(tf.math.argmax(tf.identity(y_pred), axis=0), 1)
        condition = tf.greater(tf.convert_to_tensor(puzzle), 0)
        if not is_training:
            with tf.Session() as sess: 
                print('puzzle: \n', puzzle)
                print('pred_puzzle: \n', pred.eval())
                print('pred_puzzle_indices: \n', pred[blank_indices])

        return K.mean(K.square(puzzle - pred))

    return loss

def dummy_loss(puzzle, is_training):
    # we expect this number in every row, column, and 3x3 subgrid
    unit_expected = sum(range(n2+1))

    def loss(y_true, y_pred):
        l_mse = tf.cast(K.mean(K.square(y_pred - y_true)), tf.float32)
        
        true = tf.math.add(tf.math.argmax(y_true, axis=0), 1)
        pred = tf.math.add(tf.math.argmax(y_pred, axis=0), 1)
        if not is_training:
            with tf.Session() as sess: 
                print('puzzle: \n', puzzle)
                print('true: \n', true.eval())
                print('pred: \n', pred.eval())

        row_sums = tf.math.reduce_sum(pred, axis=0) - 45
        loss_rows = tf.cast(K.mean(K.square(row_sums)), tf.float32)

        col_sums = tf.math.reduce_sum(pred, axis=1) - 45
        loss_cols = tf.cast(K.mean(K.square(col_sums)), tf.float32)

        if not is_training:
            with tf.Session() as sess: 
                print('row_sums: \n', row_sums.eval())
                print('col_sums: \n', col_sums.eval())

        # TODO: 3x3 subgrid loss

        # TODO: loss from differences between clues and preds
        return l_mse + loss_rows + loss_cols

    return loss

def sudoku_loss(puzzle):
    # we expect this number in every row, column, and 3x3 subgrid
    unit_expected = sum(range(n2+1))
    puzzle = puzzle.eval(session=tf.Session())
    print('puzzle: \n', puzzle)
    clue_indices = np.argwhere(puzzle != 0)
    blank_indices = np.argwhere(puzzle == 0)

    def loss(y_true, y_pred):
        #y_pred = tf.math.add(tf.math.argmax(y_pred, axis=0), 1.0)
        true = np.argmax(y_true, axis=0) + 1
        print('true: \n', true)
        pred = np.argmax(y_pred, axis=0) + 1
        print('pred: \n', pred)
        loss_mean_squared = K.mean(K.square(pred - true))

        pred_cols = np.sum(pred, axis=0)
        loss_cols_constraint = K.mean(K.square(pred_cols - unit_expected))

        pred_rows = np.sum(pred, axis=1)
        loss_rows_constraint = K.mean(K.square(pred_rows - unit_expected))

        pred_boxes = np.array([np.sum(x) for x in submatrices(pred)])
        loss_boxes_constraint = K.mean(K.square(pred_boxes - unit_expected))

        pred[blank_indices] = 0
        loss_clues = K.mean(K.square(pred - puzzle))

        return loss_mean_squared  + loss_cols_constraint + loss_rows_constraint + loss_boxes_constraint + loss_clues


    return loss

def generate_model():
    # Input layer -> takes in a puzzle matrix
    input_layer = keras.Input(shape=(n2, n2))

    # Concatenate different processing layers
    concat = []
    # Processing layer 1 -> The provided clues
    concat.append(keras.layers.Lambda(lambda x: x)(input_layer))

    # Processing layer 2 -> LSTM on rows
    lstm_row = keras.layers.Bidirectional(keras.layers.LSTM(2*LSTM_output_units, return_sequences=True))(input_layer)
    lstm_row = keras.layers.Bidirectional(keras.layers.LSTM(LSTM_output_units, return_sequences=True))(lstm_row)
    concat.append(lstm_row)


    # Processing layer 3 -> LSTM on columns
    transpose = keras.layers.Permute((2, 1))(input_layer)
    lstm_col = keras.layers.Bidirectional(keras.layers.LSTM(LSTM_output_units, return_sequences=True))(transpose)
    lstm_col = keras.layers.Bidirectional(keras.layers.LSTM(2*LSTM_output_units, return_sequences=True))(lstm_col)
    concat.append(lstm_col)

    # Processing layer 4 -> LSTM on submatrices
    #transform = keras.layers.Lambda(lambda x: submats_to_rows(x))(input_layer)
    #concat.append(keras.layers.Bidirectional(keras.layers.LSTM(LSTM_output_units, return_sequences=True))(transform))

    # Concatenate each layer
    concat_layer = keras.layers.Concatenate()(concat)

    dense1 = keras.layers.Dense(n2 ** 3, activation='relu')(concat_layer)
    # Dense Layer to convert to n^6 elements
    dense = keras.layers.Dense(n2 ** 2, activation='sigmoid')(dense1)
    # Reshape to n^2-by-n^2-by-n^2
    outReshaped=keras.layers.Reshape((n2, n2, n2))(dense)
    # Softmax along last layer
    out = keras.layers.Softmax(axis=-1)(outReshaped)
    
    model = keras.models.Model(inputs=input_layer, outputs=out);
    model.compile(loss=dummy_loss(input_layer, True), optimizer='adam')
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

    print(input_loss(test_quizzes[0], False)(test_solutions[0], pred_solutions[0]))
   

if __name__ == "__main__":
    main(); print("Done")
