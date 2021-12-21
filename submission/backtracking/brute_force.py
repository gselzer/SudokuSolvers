"""
An algorithm using a brute force approach to solve Sudoku puzzleSize
Approach derived from: https://www.geeksforgeeks.org/sudoku-backtracking-7/
Author: Gabriel Selzer
"""

from data_load import load_data, get_batch_data
from hyperparams import Hyperparams as hp
import numpy as np

boxSize = hp.cellSize
N=hp.puzzleSize


def isSafe(grid, row, col, num):
    for x in range(N):
        if grid[row][x] == num:
            return False
    for x in range(N):
        if grid[x][col] == num:
            return False
    startRow = row - row % boxSize
    startCol = col - col % boxSize
    for i in range(boxSize):
        for j in range(boxSize):
            if grid[i + startRow][j + startCol] == num:
                return False
    return True

def solveSudoku(grid, row, col):
    if (row == N - 1 and col == N):
        return True
    if col == N:
        row += 1
        col = 0
    if grid[row][col] > 0:
        return solveSudoku(grid, row, col + 1)
    for num in range(1, N+1, 1):
        if isSafe(grid, row, col, num):
            grid[row][col]=num
            if solveSudoku(grid, row, col + 1):
                return True
        grid[row][col] = 0
    return False

def backtrack(X):
    mutable = np.copy(X)
    if solveSudoku(mutable, 0, 0):
        return mutable
    raise ValueError(X, " has no solution!")


def main():

    X, Y = load_data(type="test")
    try:
        solution = backtrack(X[1])
    except ValueError:
        print(X, ' has no solution!')

    print(solution - Y[1])

if __name__ == "__main__":
    main()
    print("Done")
