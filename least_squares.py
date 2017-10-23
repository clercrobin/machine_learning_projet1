# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """Implements the least squares method"""
    x2 = np.dot(np.transpose(tx), tx)
    xy = np.dot(np.transpose(tx), y)
    w_star = np.linalg.solve(x2, xy)

    e = y-tx@w_star
    return np.sqrt(1/(y.shape[0])*e.transpose()@e), w_star