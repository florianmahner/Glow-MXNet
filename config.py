class Config:
    ''' Network parameters '''
    K = 10 # Number of Flow steps
    L = 2 # Number of Glow Blocks
    filter_size = 256  # hidden layer filter size of affine coupling block net
    affine = True # affine transformatiojn
    temp = 0.7 # temperature to sample gaussian from
    n_bits = 8  # 256 pixels

    ''' Training parameters '''
    batch_size = 8
    lr = 0.0002 # for adam optimizer
    n_epochs = 250
    log_dir = './logs' # log dir for summary writer to generate samples
    plot_interval = 5 # plot every x epoch a sample
    use_gpu = 1 # -1 is cpu