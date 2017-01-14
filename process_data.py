import data_utils


def LoadAndProcessData(input_columns, mode, save=False, plot=False):
    Engy, F0, y = data_utils.LoadData(mode)
    Engy, F0 = data_utils.IgnoreLowEnergyFrequence(Engy, F0)
    Engy, F0 = data_utils.TrimData(Engy, F0)
    F0 = data_utils.TransformToMelFrequencyScale(F0)
    F0 = data_utils.DivSingleDataStd(F0)
    F0 = data_utils.DivDataStd(F0)
    F0 = data_utils.SmoothF0(F0)
    F0 = data_utils.CenterlizeData(F0)
    # data_utils.PlotF0(mode, F0, y)
    Engy, F0 = data_utils.NormalizeDataLengthWithInterpolation(Engy, F0, result_len=input_columns)
    if plot:
        data_utils.PlotF0(mode, F0, y)
    if save:
        data_utils.SaveData(Engy, F0, y, mode)
    return Engy, F0, y


def LoadAndProcessTrainData(input_columns, save=False, plot=False):
    return LoadAndProcessData(input_columns, 'train', save, plot)


def LoadAndProcessValData(input_columns, save=False, plot=False):
    return LoadAndProcessData(input_columns, 'val', save, plot)

def LoadAndProcessTestData(input_columns, save=False, plot=False):
    return LoadAndProcessData(input_columns, 'test', save, plot)

def ProcessAndSaveData():
    _ = LoadAndProcessTrainData(120, True, True)
    _ = LoadAndProcessValData(120, True, True)
    _ = LoadAndProcessTestData(120, True, True)