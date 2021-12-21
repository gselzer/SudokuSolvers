from CGA import CGA
import numpy as np

"""
A pytest test used to benchmark the CGA algorithm
Author: Gabriel Selzer
"""

def load_data(f):
    npzfile = np.load(f)
    quizzes = npzfile['quizzes']
    solutions = npzfile['solutions']
    print("Loaded ", len(quizzes), " puzzles")
    print(quizzes)
    return (quizzes, solutions)

def runAll(quizzes, solutions):
    for i in range(len(quizzes)):
        result = CGA(quizzes[i])

def test_CGA(benchmark):
    quizzes, solutions = load_data('../CNN/data/sudoku.npz')
    benchmark.pedantic(runAll, kwargs={'quizzes': quizzes, 'solutions': solutions}, iterations=10)


if __name__ == "__main__":
    quizzes, solutions = load_data('../CNN/data/sudoku.npz')
    runAll(quizzes, solutions)
    print("Done")
