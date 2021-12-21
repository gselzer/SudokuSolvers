import tensorflow as tf
from tensorflow import keras
#from keras.models import Model
from hyperparams import Hyperparams as hp
from generate_sudoku import run
import numpy as np
import os.path
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

n2 = hp.puzzleSize
LSTM_output_units = 9
batch_size = hp.batch_size

#train_data = 'data/debug_n100.npz'
train_data = '../CNN/data/test100_continual.npz'

checkpoint_filename = './models/CNN_LSTM - model.h5'

def load_data(filename):
    npzfile = np.load(filename)
    return npzfile['quizzes'], npzfile['solutions']

def model_input(input):
  return np.expand_dims(input,axis=0)
  
def generatePuzzle():
    all_results, solution = run(n=0, iter=0)
    quiz = best(all_results)

def main():
    quizzes, solutions = load_data(train_data)
    solutions = solutions - 1

    if os.path.isfile(checkpoint_filename):
        print('======LOADING MODEL======')
        model = keras.models.load_model(checkpoint_filename)
        print(model.summary())
    else:
        print('Unable to find a model to visualize')
        return
    

    # Loading puzzles
    puzzle_data_save_name = 'debug_n100.npz'
    path = '../CNN/data/test100_continual.npz'
    puzzle_data = np.load(path)
    quizzes, solutions = puzzle_data['quizzes'], puzzle_data['solutions']

    test_index = 0
    example_puzzle = model_input(quizzes[test_index])
    example_solution = solutions[test_index]
    prediction = model.predict(example_puzzle)
    argmaxed = np.argmax(prediction, axis = -1) + 1
    pred = np.reshape(argmaxed, (9, 9))
    print("Example puzzle:")
    print(example_puzzle)
    print("Prediction:")
    print(pred)
    print("Solution:")
    print(example_solution)
    print("Incorrect blanks: " + str(np.count_nonzero(pred - example_solution)))


    # Creating a subsection of the model
    c1_net = keras.models.Model(inputs = model.inputs, outputs = model.layers[9].output)
    c1_net.summary()

    plt.rcParams["figure.figsize"] = [32, 32]

    feature_maps = c1_net.predict(example_puzzle)
    #filters, biases = model.layers[9].get_weights()

    print(feature_maps.shape)
    #plt.matshow(feature_maps[0, :, :, 0], cmap=plt.cm.Blues)


    # functions to plot kernels for the convolutions - Currently unused
    def plot_conv_kernel(puzzle, i, feature_maps, filters, biases):
        fig, axs = plt.subplots(1, 3)

            # specify subplot and turn of axis
        #ax = pyplot.subplot(square, square, ix)
        #pyplot.subplot(gs1[i])
        axs[0].matshow(puzzle[0], cmap=plt.cm.Blues)
        for x in range(9):
            for y in range(9):
                axs[0].text(x, y, "{:.2f}".format(puzzle[0, y, x]), va='center', ha='center')

        axs[1].set_xticks([])
        axs[1].set_yticks([])

            # plot filter channel in grayscale
        axs[1].matshow(feature_maps[0, :, :, i], cmap=plt.cm.Blues)
        for x in range(9):
            for y in range(9):
                axs[1].text(x, y, "{:.2f}".format(feature_maps[0, y, x, i]), va='center', ha='center')

        axs[2].matshow(filters[:, :, 0, i], cmap=plt.cm.Blues)
        for x in range(3):
            for y in range(3):
                axs[2].text(x, y, str(filters[y, x, 0, i]), va='center', ha='center')
        print('Kernel Bias: ' + str(biases[i]))
        # show the figure
        plt.show()

    def plot_conv1_kernel(puzzle, i):
        feature_maps = c1_net.predict(puzzle)
        filters, biases = model.layers[9].get_weights()
        plot_conv_kernel(puzzle, i, feature_maps, filters, biases)
   

if __name__ == "__main__":
    main(); print("Done")