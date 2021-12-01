import numpy as np

def unmulticlass(y, label):
    r = np.zeros_like(y)
    r[y != label] = -1
    r[y == label] = 1
    return r
