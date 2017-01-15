import data_utils

data_sets = [
    'best_data', 'divide_data', 'new_data', 'smooth_data', 'original_data'
]

data_filenames = [
    'train_f0s', 'test_f0s', 'val_f0s'
]

def ImportY(filename):
    data_file = open(filename, "r")
    data = []
    for line in data_file.readlines():
        data.append(int(line))
    return data

y_train = ImportY('train_labels')
y_test = ImportY('test_labels')
y_val = ImportY('val_labels')

data_root = 'Torch'

def LoadData(fd):
    res = []
    for line in fd.readlines():
        res.append(map(float, line.split()))

    return res

for data_set in data_sets:
    data_set_path = "%s/%s" % (data_root, data_set)
    train_file = open("%s/train_f0s" % data_set_path, "r")
    val_file = open("%s/val_f0s" % data_set_path, "r")
    test_file = open("%s/test_f0s" % data_set_path, "r")

    train_F0 = LoadData(train_file)
    val_F0 = LoadData(val_file)
    test_F0 = LoadData(test_file)

    data_utils.PlotAndSaveF0('%s-%s' % (data_set, 'train'), train_F0, y_train)
    data_utils.PlotAndSaveF0('%s-%s' % (data_set, 'val'), val_F0, y_val)
    data_utils.PlotAndSaveF0('%s-%s' % (data_set, 'test'), test_F0, y_test)