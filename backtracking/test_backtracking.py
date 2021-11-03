
from brute_force import backtrack
import numpy as np
import timeit

def load_data(f):
    npzfile = np.load(f)
    quizzes = npzfile['quizzes']
    solutions = npzfile['solutions']
    print("Loaded ", len(quizzes), " puzzles")
    return (quizzes, solutions)

def runAll(quizzes, solutions):
    starttime = timeit.default_timer()
    for i in range(len(quizzes)):
        result = backtrack(quizzes[i])

def test_backtracking(benchmark):
    quizzes, solutions = load_data('../CNN/data/sudoku.npz')
    benchmark.pedantic(runAll, kwargs={'quizzes': quizzes, 'solutions': solutions}, iterations=10)


if __name__ == "__main__":
    quizzes, solutions = load_data('../CNN/data/sudoku.npz')
    runAll(quizzes, solutions)
    print("Done")
