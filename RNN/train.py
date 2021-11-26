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

special_value = -10;
LSTM_output_units = 100;

def generate_model():
    model = keras.Sequential()
    model.add(keras.layers.Masking(mask_value=special_value, input_shape=(hp.puzzleSize, hp.puzzleSize)))
    
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(LSTM_output_units)))

    model.add(keras.layers.Dense(hp.puzzleSize ** 3, activation='softmax'))
    model.add(keras.layers.Reshape((hp.puzzleSize, hp.puzzleSize, hp.puzzleSize)))
    model.add(keras.layers.Softmax(axis=-1))

    
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(model.summary())
    return model



def main():
    model = generate_model()

if __name__ == "__main__":
    main(); print("Done")
