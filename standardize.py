import numpy as np

def standardize(x, myMean = None, myStd = None):
    ''' fill your code in here...
    '''
    if not myMean:
        myMean = np.mean(x, axis=0)
    centered_data = x - myMean
    if not myStd:
        myStd = np.std(centered_data, axis=0)
    
    std_data = centered_data / myStd
    
    return std_data, myMean, myStd