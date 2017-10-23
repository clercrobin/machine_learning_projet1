import numpy as np

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    totalLength = y.shape[0]
    intervalLength = int(totalLength / k_fold) # Length of an internval
    np.random.seed(seed)
    indices = np.random.permutation(totalLength)
    k_indices = [indices[k * intervalLength: (k + 1) * intervalLength] for k in range(k_fold)]
    #print(k_indices)
    return np.array(k_indices)