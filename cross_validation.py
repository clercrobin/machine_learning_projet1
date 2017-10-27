import numpy as np
from implementations import ridge_regression, compute_mse

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    p = np.random.permutation(len(x))
    x = x[p]
    y = y[p]
    n_train = int(ratio * len(x))
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]