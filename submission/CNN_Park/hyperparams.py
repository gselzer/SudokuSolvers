class Hyperparams:
    '''Hyper parameters'''
    # data
    train_fpath = './data/sudoku.npz'
    test_fpath = './data/test_n100easy.npz'
    
    # model
    num_blocks = 10
    num_filters = 512
    filter_size = 3
    
    # training scheme
    lr = 0.0001
    logdir = "logdir"
    batch_size = 512
    num_epochs = 3
    cellSize = 3
    puzzleSize = cellSize ** 2

