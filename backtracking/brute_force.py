# Taken from: https://www.geeksforgeeks.org/sudoku-backtracking-7/
# Author: Gabriel Selzer

from data_load import load_data, get_batch_data

boxSize = 3
N=boxSize * boxSize


def isSafe(grid, row, col, num):
    for x in range(9):
        if grid[row][x] == num:
            return False
    for x in range(9):
        if grid[x][col] == num:
            return False
    startRow = row - row % 3
    startCol = col - col % 3
    for i in range(3):
        for j in range(3):
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

def main():
    
    X, Y = load_data(type="test")
    if solveSudoku(X[1], 0, 0):
        print(X[1] - Y[1])

if __name__ == "__main__":
    main(); print("Done")
