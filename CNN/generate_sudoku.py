#!/usr/bin/python2
"""
This is adapted from https://www.ocf.berkeley.edu/~arel/sudoku/main.html.
Generates 1 million Sudoku games. 
Kyubyong Park. kbpark.linguist@gmail.com www.github.com/kyubyong
"""

import random, copy
import numpy as np
from hyperparams import Hyperparams as hp
import sys

verbose = False
num = 10

sample  = [ [3,4,1,2,9,7,6,8,5],
            [2,5,6,8,3,4,9,7,1],
            [9,8,7,1,5,6,3,2,4],
            [1,9,2,6,7,5,8,4,3],
            [8,7,5,4,2,3,1,9,6],
            [6,3,4,9,1,8,2,5,7],
            [5,6,3,7,8,9,4,1,2],
            [4,1,9,5,6,2,7,3,8],
            [7,2,8,3,4,1,5,6,9] ]
            
"""
Randomly arrange numbers in a grid while making all rows, columns and
squares (sub-grids) contain the numbers 1 through 9.

For example, "sample" (above) could be the output of this function. """
def construct_puzzle_solution():
    base  = hp.cellSize
    side  = base*base

    # pattern for a baseline valid solution
    def pattern(r,c): return (base*(r%base)+r//base+c)%side

    # randomize rows, columns and numbers (of valid base pattern)
    from random import sample
    def shuffle(s): return sample(s,len(s)) 
    rBase = range(base) 
    rows  = [ g*base + r for g in shuffle(rBase) for r in shuffle(rBase) ] 
    cols  = [ g*base + c for g in shuffle(rBase) for c in shuffle(rBase) ]
    nums  = shuffle(range(1,base*base+1))

    # produce board using randomized baseline pattern
    board = [ [nums[pattern(r,c)] for c in cols] for r in rows ]
    if (verbose):
        print(np.array(board))
    return board


"""
Randomly pluck out cells (numbers) from the solved puzzle grid, ensuring that any
plucked number can still be deduced from the remaining cells.

For deduction to be possible, each other cell in the plucked number's row, column,
or square must not be able to contain that number. """
def pluck(puzzle, n=0):

    """
    Answers the question: can the cell (i,j) in the puzzle "puz" contain the number
    in cell "c"? """
    def canBeA(puz, i, j, c):
        v = puz[int(c/hp.puzzleSize)][c%hp.puzzleSize]
        if puz[i][j] == v: return True
        if puz[i][j] in range(1,hp.puzzleSize+1): return False
            
        for m in range(hp.puzzleSize): # test row, col, square
            # if not the cell itself, and the mth cell of the group contains the value v, then "no"
            if not (m==c/hp.puzzleSize and j==c%hp.puzzleSize) and puz[m][j] == v: return False
            if not (i==c/hp.puzzleSize and m==c%hp.puzzleSize) and puz[i][m] == v: return False
            if not ((i/hp.cellSize)*hp.cellSize + m/hp.cellSize==c/hp.puzzleSize and (j/hp.cellSize)*hp.cellSize + m%hp.cellSize==c%hp.puzzleSize) and puz[(i/hp.cellSize)*hp.cellSize + m/hp.cellSize][(j/hp.cellSize)*hp.cellSize + m%hp.cellSize] == v:
                return False

        return True


    """
    starts with a set of all 81 cells, and tries to remove one (randomly) at a time
    but not before checking that the cell can still be deduced from the remaining cells. """
    cells     = set(range(hp.puzzleSize ** 2))
    cellsleft = cells.copy()
    while len(cells) > n and len(cellsleft):
        cell = int(random.choice(list(cellsleft))) # choose a cell from ones we haven't tried
        cellsleft.discard(cell) # record that we are trying this cell

        # row, col and square record whether another cell in those groups could also take
        # on the value we are trying to pluck. (If another cell can, then we can't use the
        # group to deduce this value.) If all three groups are True, then we cannot pluck
        # this cell and must try another one.
        row = col = square = False

        for i in range(hp.puzzleSize):
            if i != cell/hp.puzzleSize:
                j = cell % hp.puzzleSize
                if canBeA(puzzle, i, j, cell): row = True
            if i != cell%hp.puzzleSize:
                i = int(cell / hp.puzzleSize)
                if canBeA(puzzle, i, i, cell): col = True
            if not (((cell/hp.puzzleSize)/hp.cellSize)*hp.cellSize + i/hp.cellSize == cell/hp.puzzleSize and ((cell/hp.puzzleSize)%hp.cellSize)*hp.cellSize + i%hp.cellSize == cell%hp.puzzleSize):
                if canBeA(puzzle, ((cell/hp.puzzleSize)/hp.cellSize)*hp.cellSize + i/hp.cellSize, ((cell/hp.puzzleSize)%hp.cellSize)*hp.cellSize + i%hp.cellSize, cell): square = True

        if row and col and square:
            continue # could not pluck this cell, try again.
        else:
            # this is a pluckable cell!
            i = int(cell / hp.puzzleSize)
            puzzle[i][cell%hp.puzzleSize] = 0 # 0 denotes a blank cell
            cells.discard(cell) # remove from the set of visible cells (pluck it)
            # we don't need to reset "cellsleft" because if a cell was not pluckable
            # earlier, then it will still not be pluckable now (with less information
            # on the board).

    # This is the puzzle we found, in all its glory.
    return (puzzle, len(cells))
    
    
"""
That's it.

If we want to make a puzzle we can do this:
    pluck(construct_puzzle_solution())
    
The following functions are convenience functions for doing just that...
"""



"""
This uses the above functions to create a new puzzle. It attempts to
create one with 28 (by default) given cells, but if it can't, it returns
one with as few givens as it is able to find.

This function actually tries making 100 puzzles (by default) and returns
all of them. The "best" function that follows this one selects the best
one of those.
"""
def run(n = 28, iter=100):
    all_results = {}
    if (verbose):
        print("Constructing a sudoku puzzle.")
        print("* creating the solution...")
    a_puzzle_solution = construct_puzzle_solution()
    
    if (verbose):
        print("* constructing a puzzle...")
    for i in range(iter):
        puzzle = copy.deepcopy(a_puzzle_solution)
        (result, number_of_cells) = pluck(puzzle, n)
        all_results.setdefault(number_of_cells, []).append(result)
        if number_of_cells <= n: break
 
    return all_results, a_puzzle_solution

def best(set_of_puzzles):
    # Could run some evaluation function here. For now just pick
    # the one with the fewest "givens".
    return set_of_puzzles[min(set_of_puzzles.keys())][0]

def display(puzzle):
    for row in puzzle:
        print(' '.join([str(n or '_') for n in row]))

    
# """ Controls starts here """
# results = run(n=0)       # find puzzles with as few givens as possible.
# puzzle  = best(results)  # use the best one of those puzzles.
# display(puzzle)          # display that puzzle.


def main(num, fill):
    '''
    Generates `num` games of Sudoku.
    '''
    quizzes = np.zeros((num, hp.puzzleSize, hp.puzzleSize), np.int32)
    solutions = np.zeros((num, hp.puzzleSize, hp.puzzleSize), np.int32)
    for i in range(num):
        all_results, solution = run(n=fill, iter=11)
        quiz = best(all_results)
        
        quizzes[i] = quiz
        solutions[i] = solution

        # save every 10 puzzles
        if (i+1) % (10) == 0:
            np.savez('data/sudoku.npz', quizzes=quizzes, solutions=solutions)

        if (i+1) % (100) == 0:
            print("Puzzle " + str(i+1) + " of " + str(num))

def parse():
    for arg in sys.argv:
        if (arg == "-v"):
            global verbose
            verbose = True
        if (arg.startswith("-num")):
            global num
            num = int(arg.split("=")[1])
            print("Generating " + str(num) + " puzzles")    
        if (arg.startswith("-fill")):
            global fill
            fill = int(arg.split("=")[1])
            print("Number of filled puzzle cells: " + str(fill))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        parse()
    main(num, fill)
    print("Done!")
