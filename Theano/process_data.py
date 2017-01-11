import data_utils


def LoadAndProcessData(input_columns, mode, save=False):
    Engy, F0, y = data_utils.LoadData(mode)
    Engy, F0 = data_utils.IgnoreLowEnergyFrequence(Engy, F0)
    Engy, F0 = data_utils.TrimData(Engy, F0)
    Engy, F0 = data_utils.LogData(Engy, F0)
    Engy, F0 = data_utils.NormalizeDataLengthWithInterpolation(Engy, F0, result_len=input_columns)
    if save:
        data_utils.SaveData(Engy, F0, y, mode)
    return Engy, F0, y


def LoadAndProcessTrainData(input_columns, save=False):
    return LoadAndProcessData(input_columns, 'train', save)


def LoadAndProcessValData(input_columns, save=False):
    return LoadAndProcessData(input_columns, 'val', save)

def LoadAndProcessTestData(input_columns, save=False):
    return LoadAndProcessData(input_columns, 'test', save)