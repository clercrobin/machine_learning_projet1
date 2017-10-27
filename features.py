import numpy as np

def standardize(x, mean=None, std=None):
    mean = np.mean(x, axis=0) if mean is None else mean
    centered_data = x - mean
    std = np.std(centered_data, axis=0) if std is None else std
    std[std == 0] = 1
    std_data = centered_data / std
    return std_data, mean, std

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    
    poly_basis = np.column_stack([np.power(x, i) for i in range(1,degree+1)])

    # Cross terms
    indices_i = np.array(np.sum([[i for j in range(i)] for i in range(x.shape[1])]))
    indices_j = np.array(np.sum([[j for j in range(i)] for i in range(x.shape[1])]))
    cross_terms = x[:,indices_i] * x[:,indices_j]
    x = np.hstack((poly_basis, cross_terms))
    
    # Add a bias term
    x = np.c_[np.ones(len(x)), x]
    
    return x