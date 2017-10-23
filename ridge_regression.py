import numpy as np
from costs import compute_loss


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    w = np.dot(np.linalg.inv(np.dot(tx.T, tx)+lambda_*2*y.shape[0]*np.eye(tx.shape[1])), np.dot(tx.T, y))
    mse = compute_loss(y,tx,w)
    return mse, w