import os
import copy
import scipy.interpolate as spi
import numpy

data_root = 'toneclassifier'
train_data_path = "%s/train" % data_root
val_data_path = "%s/test" % data_root
test_data_path = "%s/test_new" % data_root

labels = {
    'one': 0,
    'two': 1,
    'three': 2,
    'four': 3
}

def LoadData(mode='train'):
    data_path = train_data_path
    if mode == 'val':
        data_path = val_data_path
    elif mode == 'test':
        data_path = test_data_path
    data_len_max = 0
    Engy = []
    F0 = []
    y = []
    for labelName, label in labels.iteritems():
        data_subset_path = "%s/%s" % (data_path, labelName)
        data_names = set()
        for filename in os.listdir(data_subset_path):
            if filename[0] == ".":
                continue
            if ".engy" in filename:
                data_names.add(filename[0:-5])
            elif ".f0" in filename:
                data_names.add(filename[0:-3])

        for data_name in data_names:
            engy = map(float, open("%s/%s.engy" % (data_subset_path, data_name)).readlines())
            f0 = map(float, open("%s/%s.f0" % (data_subset_path, data_name)).readlines())
            if data_len_max < len(engy):
                data_len_max = len(engy)
            Engy.append(engy)
            F0.append(f0)
            y.append(label)
    return Engy, F0, y


def IgnoreLowEnergyFrequence(Engy, F0):
    data_num = len(Engy)
    if data_num != len(F0):
        raise ValueError("the number of input data mismatched. len(Engy)==%d and len(F0)==%d" % (len(Engy), len(F0)))

    resEngy = []
    resF0 = []

    for i in xrange(data_num):
        engy = copy.copy(Engy[i])
        f0 = copy.copy(F0[i])
        data_len = len(engy)
        if data_len != len(f0):
            raise ValueError("the length of %d-th data mismatched. len(engy)==%d and len(f0)==%d" % (i, len(engy), len(f0)))

        zero_freq_engy_sum = 0.0
        zero_freq_count = 0.0
        for j in xrange(data_len):
            if f0[j] < 1e-4:
                zero_freq_count += 1
                zero_freq_engy_sum += engy[j]

        mean_engy = zero_freq_engy_sum / zero_freq_count
        for j in xrange(data_len):
            if engy[j] <= mean_engy:
                f0[j] = 0.0

        resEngy.append(engy)
        resF0.append(f0)

    return resEngy, resF0


def TrimData(Engy, F0):
    data_num = len(Engy)
    if data_num != len(F0):
        raise ValueError("the number of input data mismatched. len(Engy)==%d and len(F0)==%d" % (len(Engy), len(F0)))

    resEngy = []
    resF0 = []

    for i in xrange(data_num):
        engy = copy.copy(Engy[i])
        f0 = copy.copy(F0[i])
        data_len = len(engy)
        if data_len != len(f0):
            raise ValueError("the length of %d-th data mismatched. len(engy)==%d and len(f0)==%d" % (i, len(engy), len(f0)))

        start = None
        end = None

        for i in xrange(len(f0)):
            if f0[i] > 1e-5:
                start = i
                break
        for i in xrange(len(f0) - 1, -1, -1):
            if f0[i] > 1e-5:
                end = i + 1
                break

        resEngy.append(copy.copy(engy[start:end]))
        resF0.append(copy.copy(f0[start:end]))
    return resEngy, resF0

def NormalizeDataLengthWithInterpolation(Engy, F0, result_len=200):
    data_num = len(Engy)
    if data_num != len(F0):
        raise ValueError("the number of input data mismatched. len(Engy)==%d and len(F0)==%d" % (len(Engy), len(F0)))

    resEngy = []
    resF0 = []

    for i in xrange(data_num):
        engy = copy.copy(Engy[i])
        f0 = copy.copy(F0[i])
        data_len = len(engy)
        if data_len != len(f0):
            raise ValueError(
                "the length of %d-th data mismatched. len(engy)==%d and len(f0)==%d" % (i, len(engy), len(f0)))
        k = float(result_len - 1) / float(data_len - 1)
        x = [i * k for i in xrange(data_len)]
        newX = [i * 1.0 for i in xrange(result_len)]
        newX[-1] = x[-1]
        new_engy = spi.interp1d(x, engy, kind='cubic')(newX)
        new_f0 = spi.interp1d(x, f0, kind='cubic')(newX)

        resEngy.append(new_engy)
        resF0.append(new_f0)

    return resEngy, resF0


def SaveData(Engy, F0, y, mode='train'):
    save_engy_name = 'train_engys'
    save_f0_name = 'train_f0s'
    save_y_name = 'train_labels'
    if mode == 'val':
        save_engy_name = 'val_engys'
        save_f0_name = 'val_f0s'
        save_y_name = 'val_labels'
    elif mode == 'test':
        save_engy_name = 'test_engys'
        save_f0_name = 'test_f0s'
        save_y_name = 'test_labels'

    engy_file = open(save_engy_name, "w")
    f0_file = open(save_f0_name, "w")
    y_file = open(save_y_name, "w")

    data_num = len(Engy)
    if data_num != len(F0) or data_num != len(y):
        raise ValueError("the number of data mismatched, Engy:%d, F0:%d, y:%d" % (len(Engy), len(F0), len(y)))

    for i in xrange(data_num):
        engy_file.write("%s\n" % (' '.join(map(lambda x: "%.5f" % x, Engy[i]))))
        f0_file.write("%s\n" % (' '.join(map(lambda x: "%.5f" % x, F0[i]))))
        y_file.write("%d\n"% y[i])

    engy_file.close()
    f0_file.close()
    y_file.close()

