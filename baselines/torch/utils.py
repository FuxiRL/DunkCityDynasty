import random
import numpy as np
import torch

def onehot(num, size):
    """ One-hot encoding
    """
    onehot_vec = np.zeros(size)
    if num < size:
        onehot_vec[num] = 1
    return onehot_vec.tolist()

def all_seed(seed = 0):
    ''' Set all seeds to the same value
    '''
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

