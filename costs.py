# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np

def compute_loss(y, tx, w):
    """Computes the MSE loss"""
    e = y-tx@w
    return 1/(2*y.shape[0])*e.transpose()@e

def compute_mse(y, tx, w):
    """Computes the MSE loss"""
    e = y-tx@w
    return 1/(2*y.shape[0])*e.transpose()@e

def compute_rmse(y, tx, w):
    """Computes the RMSE loss"""
    return np.sqrt(2*compute_mse(y,tx,w))

def rmse_from_mse (mse):
    """Computes the RMSE loss from the MSE loss"""
    return np.sqrt(2*mse)