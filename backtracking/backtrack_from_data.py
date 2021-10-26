
from brute_force import backtrack
import numpy as np

def load_data(f):
    npzfile = np.load(f)
    quizzes = npzfile['quizzes']
    solutions = npzfile['solutions']
    return (quizzes, solutions)

def runAll():
    quizzes, solutions = load_data('../CNN/data/sudoku.npz')
    print(quizzes)
    for i in range(len(quizzes)):
        result = backtrack(quizzes[i])
        print("Expected - Actual: ", solutions[i] - result)


if __name__ == "__main__":
    runAll()
    print("Done")
