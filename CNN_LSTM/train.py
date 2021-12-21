# -*- coding: utf-8 -*-
#/usr/bin/python2

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
train_data = '../CNN/data/test100000_continual.npz'

def load_data(filename):
    npzfile = np.load(filename)
    return npzfile['quizzes'], npzfile['solutions']

def generate_model():
    # Input layer -> takes in a puzzle matrix
    input_layer = keras.Input(shape=(n2, n2))
    input_reshape = keras.layers.Reshape((n2, n2, 1))(input_layer)
    input_norm = keras.layers.BatchNormalization()(input_layer)

    concat = []
    # Processing layer 1 -> The provided clues
    concat.append(keras.layers.Lambda(lambda x: x)(input_norm))

    # Processing layer 2 -> LSTM on rows
    lstm_row = keras.layers.Bidirectional(keras.layers.LSTM(2*LSTM_output_units, return_sequences=True))(input_norm)
    lstm_row = keras.layers.Bidirectional(keras.layers.LSTM(LSTM_output_units, return_sequences=True))(lstm_row)
    concat.append(lstm_row)


    # Processing layer 3 -> LSTM on columns
    transpose = keras.layers.Permute((2, 1))(input_norm)
    lstm_col = keras.layers.Bidirectional(keras.layers.LSTM(10 * LSTM_output_units, return_sequences=True))(transpose)
    lstm_col = keras.layers.Bidirectional(keras.layers.LSTM(10*LSTM_output_units, return_sequences=True))(lstm_col)
    lstm_col = keras.layers.Bidirectional(keras.layers.LSTM(2*LSTM_output_units, return_sequences=True))(lstm_col)
    concat.append(lstm_col)

    # Processing layer 4 -> LSTM on submatrices
    #transform = keras.layers.Lambda(lambda x: submats_to_rows(x))(input_layer)
    #concat.append(keras.layers.Bidirectional(keras.layers.LSTM(LSTM_output_units, return_sequences=True))(transform))

    # Concatenate each layer
    concat_layer = keras.layers.Concatenate()(concat)
    concat_reshape = keras.layers.Reshape((n2, n2, 7))(concat_layer)
    concat_norm = keras.layers.BatchNormalization()(concat_reshape)

    conv1 = keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(concat_norm)
    norm1 = keras.layers.BatchNormalization()(conv1)

    conv2 = keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(norm1)
    norm2 = keras.layers.BatchNormalization()(conv2)

    conv3 = keras.layers.Conv2D(128, kernel_size=(1,1), activation='relu', padding='same')(norm2)
    norm3 = keras.layers.BatchNormalization()(conv3)

    flat = keras.layers.Flatten()(norm3)
    dense = keras.layers.Dense(n2 ** 3)(flat)

    out_reshape = keras.layers.Reshape((-1, n2))(dense)
    act = keras.layers.Activation('softmax')(out_reshape)

    model = keras.models.Model(inputs=input_layer, outputs=act)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    print(model.summary())
    return model

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

checkpoint_filepath = './models/'
checkpoint_filename = checkpoint_filepath + 'CNN_LSTM - model.h5'
def training_callbacks():
    if not os.path.exists(checkpoint_filepath):
        os.makedirs(checkpoint_filepath)
    checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_filename, monitor='loss', verbose=1, save_best_only=True)
    return [checkpoint]

def main():
    quizzes, solutions = load_data(train_data)
    solutions = solutions - 1

    for i in range(20):
        print(quizzes[i])

    if os.path.isfile(checkpoint_filename):
        print('======LOADING MODEL======')
        model = keras.models.load_model(checkpoint_filename)
        print(model.summary())
    else:
        print('======TRAINING MODEL======')
        model = generate_model()

        model.fit(quizzes, solutions, batch_size=batch_size, epochs=hp.num_epochs, verbose=1, callbacks = training_callbacks())
   

if __name__ == "__main__":
    main(); print("Done")
