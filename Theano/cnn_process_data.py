import process_data
import numpy as np

def ImportData(input_columns, mode='train'):
    engy, f0, y = process_data.LoadAndProcessData(input_columns, mode)
    X = [engy, f0]
    X = np.asarray(X)
    X = X.swapaxes(0, 1)
    return X, y

