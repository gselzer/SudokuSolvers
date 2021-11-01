# Taken from: https://www.geeksforgeeks.org/sudoku-backtracking-7/
# Author: Gabriel Selzer

from data_load import load_data, get_batch_data
from hyperparams import Hyperparams as hp
import numpy as np
import math
import random

boxSize = hp.cellSize
N=boxSize * boxSize

boxElements = set(range(1, N + 1))


# Populates a NxN subgrid with the digits [1, N^2] not currently in the subgrid.
def populateSubgrid(grid, r, c):
    # get subgrid
    cstart = c * boxSize;
    rstart = r * boxSize;
    subgrid = grid[cstart : cstart + boxSize, rstart : rstart + boxSize]
    # get a randomly shuffled list of the missing numbers
    nums = range(1, N+1)
    for x in np.unique(subgrid):
        if x != 0:
            nums.remove(x)
    random.shuffle(nums)
    numsInd = 0;
    for i in range(subgrid.shape[0]):
        for j in range(subgrid.shape[1]):
            if subgrid[i][j] == 0:
                subgrid[i][j] = nums[numsInd]
                numsInd = numsInd+1

# Populates a grid randomly such that each NxN subgrid contains each number [1, N^2] exactly once
def populateGrid(grid):
    copy = np.matrix.copy(grid)
    for i in range(0, boxSize):
        for j in range(0, boxSize):
            populateSubgrid(copy, i, j)
    return copy;

# determines the number of digits in the range [1, N^2] not in row r of grid grid
def rowFitness(grid, r):
    l = boxElements.difference(np.unique(grid[r, :]))
    return len(l)

# determines the number of digits in the range [1, N^2] not in column c of grid grid
def colFitness(grid, c):
    l = boxElements.difference(np.unique(grid[:, c]))
    return len(l)

# returns the fitness of grid g, defined as the sum of the rowfitness of each row in g, plus the colfitness of each column in g
def fitness(grid):
    fitness = sum([rowFitness(grid, i) + colFitness(grid, i) for i in range(N)])
    return fitness

def fitnessSort(population):
    population.sort(key = lambda x: x[1])

def mutateSubgrid(grid, mask, x, y):
    cstart = x * boxSize;
    rstart = y * boxSize;
    subgrid = grid[cstart : cstart + boxSize, rstart : rstart + boxSize]
    submask = mask[cstart : cstart + boxSize, rstart : rstart + boxSize]
    #indices = np.random.choice(np.argwhere(submask == 1), 2, replace=False)
    mutableIndices = np.argwhere(submask == 0)
    if mutableIndices.size == 0:
        return
    i = mutableIndices[np.random.randint(0, mutableIndices.shape[0], 2)]
    subgrid[i[0,0], i[0, 1]], subgrid[i[1,0], i[1, 1]] = subgrid[i[1, 0], i[1, 1]], subgrid[i[0, 0], i[0, 1]]
    

def mutate(grid, mask, m):
    copy = np.matrix.copy(grid)
    for i in range(m):
        r = random.randint(0, 3)
        c = random.randint(0, 3)
        mutateSubgrid(copy, mask, r, c)
    return copy
    


def CGA(grid):
    # create mask of correct answers
    mask = np.zeros(grid.shape)
    mask[grid > 0] = 1

    # create population
    offspring = 100
    population = [populateGrid(grid) for x in range(offspring)]
    population = [(g, fitness(g)) for g in population]
    print(grid)

    # initially sort population by fitness
    fitnessSort(population)

    # breed population until we find a specimen with fitness of zero
    while (population[0][1] != 0):
        print("Best fitness: ", population[0][1])
        p = population[0][1]
        m = int(math.ceil(p))
        for i in range(offspring):
            copy = mutate(population[i][0], mask, m)
            copyFitness = fitness(copy)
            population.append((copy, copyFitness))
        #print(len(population))

        fitnessSort(population)
        #tmp = population[:50]
        #del population[:50]
        #tmp.append(random.sample(population, 50))
        #population = tmp
        #print(len(population))
        tmp = population[:10]
        population = tmp + random.sample(population, 90)
        print(len(population))

        #fitnessSort(population)
        #del population[-offspring:]

    # print(population[0][0])
    return population[0][0]



def main():
    # load matrix
    X, Y = load_data(type="test")
    grid = np.array(X[1])
    # print solution
    solution = CGA(grid)
    print(grid)


if __name__ == "__main__":
    CGA(); print("Done")
