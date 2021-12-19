# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/sudoku
'''
from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np
from hyperparams import Hyperparams as hp
import os

test_data = 'data/test_n100easy.npz'
checkpoint_filename = './model.h5'


def load_data(filename):
    npzfile = np.load(filename)
    return npzfile['quizzes'], npzfile['solutions']

def write_to_file(x, y, preds, fout):
    '''Writes to file.
    Args:
      x: A 3d array with shape of [N, 9, 9]. Quizzes where blanks are represented as 0's.
      y: A 3d array with shape of [N, 9, 9]. Solutions.
      preds: A 3d array with shape of [N, 9, 9]. Predictions.
      fout: A string. File path of the output file where the results will be written.
    '''
    with open(fout, 'w') as fout:
        total_hits, total_blanks = 0, 0
        for xx, yy, pp in zip(x.reshape(-1, hp.puzzleSize*hp.puzzleSize), y.reshape(-1, hp.puzzleSize*hp.puzzleSize), preds.reshape(-1, hp.puzzleSize*hp.puzzleSize)): # sample-wise
            fout.write("qz: {}\n".format("".join(str(num) if num != 0 else "_" for num in xx)))
            fout.write("sn: {}\n".format("".join(str(num) for num in yy)))
            fout.write("pd: {}\n".format("".join(str(num) for num in pp)))

            expected = yy[xx == 0]
            got = pp[xx == 0]

            num_hits = np.equal(expected, got).sum()
            num_blanks = len(expected)

            fout.write("accuracy = %d/%d = %.2f\n\n" % (num_hits, num_blanks, float(num_hits) / num_blanks))

            total_hits += num_hits
            total_blanks += num_blanks
        fout.write("Total accuracy = %d/%d = %.2f\n\n" % (total_hits, total_blanks, float(total_hits) / total_blanks))

def arg_max(solution):
    return np.argmax(solution, axis=-1) + 1

def predict(model, x, y, fout):
    print(x.shape)
    preds = model.predict(x)
    print(preds.shape)
    preds = np.reshape(arg_max(preds), (preds.shape[0], 9, 9))

    write_to_file(x.astype(np.int32), y, preds.astype(np.int32), fout)


def test():
    x, y = load_data(test_data)
    
    model = keras.models.load_model(checkpoint_filename)

    if not os.path.exists('results'): os.mkdir('results')
    fout = 'results.txt'
    predict(model, x, y, fout)


def test_benchmark(benchmark):
    x, y = load_data_npz(type="test")
    print(x)
    
    g = Graph(is_training=False)
    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")
            
	    if not os.path.exists('results'): os.mkdir('results')
            fout = 'results/{}.txt'
	    benchmark.pedantic(predict, kwargs={'g': g, 'sess': sess, 'x': x, 'y': y, 'fout': fout}, iterations=10)
     
if __name__ == '__main__':
    test()
    print("Done")
