from costs import compute_loss
from ridge_regression import ridge_regression
from build_polynomial import build_poly
import numpy as np


def cross_validation(y, x, k_indices, k, lambda_, degree):
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
    # form data with polynomial degree: TODO
    # ***************************************************
    x_train_extended = build_poly(x_train, degree)
    #print(x_train_extended.shape)
    x_test_extended = build_poly(x_test, degree)
    #print(x_test_extended.shape)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    loss_tr,w_tr = ridge_regression(y_train,x_train_extended,lambda_)
    #print("One new regression")
    #print(w_tr)
    loss_te = compute_loss(y_test,x_test_extended,w_tr)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # calculate the loss for train and test data: TODO
    # ***************************************************
    return loss_tr, loss_te