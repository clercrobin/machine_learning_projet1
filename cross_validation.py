import numpy as np
from implementations import ridge_regression, compute_mse, least_squares_SGD,reg_logistic_regression_SGD, logistic_regression
from proj1_helpers import *

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed()
    p = np.random.permutation(len(x))
    x = x[p]
    y = y[p]
    n_train = int(ratio * len(x))
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]

def accuracy(y, y_pred):
    """ Compute accuracy. """
    right = np.sum(y_pred == y)
    wrong = len(y_pred) - right
    accuracy = right / len(y)

    #print("Good prediction: %i/%i (%.3f%%)\nWrong prediction: %i/%i (%.3f%%)" %
        #(right, len(y), 100.0 * accuracy, wrong, len(y), 100.0 * (1-accuracy)))
    
    return accuracy

def cross_validation(y, x, k_indices, k, lambda_, gamma):
    """return the loss of ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # get k'th subgroup in test, others in train: TODO
    # ***************************************************
    all_indices = np.arange(y.shape[0])
    excepted = np.setdiff1d(all_indices,k_indices[k])
    x_test = x[k_indices[k]]
    y_test = y[k_indices[k]]
    x_train = x[excepted]
    y_train = y[excepted]
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    #w, loss = ridge_regression(y_train,x_train, lambda_)
    #w, loss = least_squares_SGD(y_train, x_train, np.zeros(x_train.shape[1]), 50, lambda_)
    #w, loss = logistic_regression(y_train, x_train, np.zeros(x_train.shape[1]), 50, lambda_)
    w, loss = reg_logistic_regression_SGD((y_train == 1).astype(float), x_train, lambda_, np.zeros(x_train.shape[1]), 2000, gamma)
    #print("One new regression")
    #print(w_tr)
    y_pred = predict_labels(w, x_test)
    
    accur = accuracy(y_test, y_pred)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # calculate the loss for train and test data: TODO
    # ***************************************************
    return accur