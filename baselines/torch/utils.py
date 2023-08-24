import numpy as np

def onehot(num, size):
    """ One-hot encoding
    """
    onehot_vec = np.zeros(size)
    if num < size:
        onehot_vec[num] = 1
    return onehot_vec.tolist()