import data_utils


def LoadAndProcessData(mode, save=False):
    Engy, F0, y = data_utils.LoadData(mode)
    Engy, F0 = data_utils.IgnoreLowEnergyFrequence(Engy, F0)
    Engy, F0 = data_utils.TrimData(Engy, F0)
    Engy, F0 = data_utils.NormalizeDataLengthWithInterpolation(Engy, F0)
    if save:
        data_utils.SaveData(Engy, F0, y, mode)
    return Engy, F0, y


def LoadTrainData(save=False):
    return LoadAndProcessData('train', save)


def LoadValData(save=False):
    return LoadAndProcessData('val', save)