def compute_log_p(X, mean, sigma):
    dxm = X - mean
    #print(dxm.shape)
    #print(np.dot(dxm, np.linalg.inv(sigma)).shape)
    exponent = -0.5 * np.sum(dxm * np.dot(dxm, np.linalg.inv(sigma)), axis=1)
    return exponent - np.log(2 * np.pi) * (d / 2) - 0.5 * np.log(np.linalg.det(sigma))