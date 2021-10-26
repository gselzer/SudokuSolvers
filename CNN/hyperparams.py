class Hyperparams:
    '''Hyper parameters'''
    # data
    train_fpath = './data/sudoku.csv'
    test_fpath = './data/test.csv'
    
    # model
    num_blocks = 10
    num_filters = 512
    filter_size = 3
    
    # training scheme
    lr = 0.0001
    logdir = "logdir"
    batch_size = 64
    num_epochs = 3
    cellSize = 4
    puzzleSize = cellSize ** 2

