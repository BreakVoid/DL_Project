from theano.scalar import int32

import process_data
import data_utils
import numpy as np

def ImportData(input_columns, mode='train'):
    engy, f0, y = process_data.LoadAndProcessData(input_columns, mode)
    X = [engy, f0]
    X = np.asarray(X)
    X = X.swapaxes(0, 1)
    n, c, w = X.shape
    X = X.reshape([n, c, w, 1])
    y = np.asarray(y, dtype=int32)
    X, y = data_utils.unison_shuffled_copies(X, y)
    return X, y



