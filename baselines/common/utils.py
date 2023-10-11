import numpy as np
import os
import random
import torch

def onehot(num, size):
    """ One-hot encoding
    """
    onehot_vec = np.zeros(size)
    if num < size:
        onehot_vec[num] = 1
    return onehot_vec.tolist()

def all_seed(seed = 1):
    ''' 设置随机种子，保证实验可复现，同时保证GPU和CPU的随机种子一致
    '''
    if seed == 0: # 值为0时不设置随机种子
        return
    os.environ['PYTHONHASHSEED'] = str(seed) # set PYTHONHASHSEED env var at fixed value
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # 
    torch.cuda.manual_seed(seed) # config for GPU
    # config for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False