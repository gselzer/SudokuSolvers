
from CGA import CGA
import numpy as np
import timeit

def load_data(f):
    npzfile = np.load(f)
    quizzes = npzfile['quizzes']
    solutions = npzfile['solutions']
    print("Loaded ", len(quizzes), " puzzles")
    return (quizzes, solutions)

def runAll():
    quizzes, solutions = load_data('../CNN/data/sudoku.npz')
    starttime = timeit.default_timer()
    for i in range(len(quizzes)):
        result = CGA(quizzes[i])
        print("Solved puzzle ", i);
    print("The time difference is: ", timeit.default_timer() - starttime)


if __name__ == "__main__":
    runAll()
    print("Done")
