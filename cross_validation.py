import numpy as np
from implementations import ridge_regression, compute_mse

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed yes because of the gender
    np.random.seed(seed)
    indexes_shuffled = np.random.shuffle(np.arange(y.shape[0]))
    threshold = round(ratio*y.shape[0])
    x_train = x[:threshold]
    x_test = x[threshold:]
    y_train = y[:threshold]
    y_test = y[threshold:]
    return x_train, y_train, x_test, y_test