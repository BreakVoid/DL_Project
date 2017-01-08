import os
import scipy.interpolate as spi

dataLabel = ["one", "two", "three", "four"]

dataRoot = "../toneclassifier/train"

normalLen = 1000

for label in dataLabel:
    subsetPath = dataRoot + "/" + label
    dataset = set()
    for filename in os.listdir(subsetPath):
        if filename[0] == ".":
            continue
        if ".engy" in filename:
            dataset.add(filename[0:-5])
        elif ".f0" in filename:
            dataset.add(filename[0:-3])
    try:
        os.makedirs("../data-process-output/trim-interpolation/train/" + label)
    except OSError as err:
        pass

    for dataname in dataset:
        engyfile = open(subsetPath + "/" + dataname + ".engy", "r")
        f0file = open(subsetPath + "/" + dataname + ".f0", "r")
        engy = map(float, engyfile.readlines())
        f0 = map(float, f0file.readlines())
        engyfile.close()
        f0file.close()

        start = None
        end = None

        for i in xrange(len(f0)):
            if (f0[i] > 1e-5):
                start = i
                break
        for i in xrange(len(f0) - 1, -1, -1):
            if (f0[i] > 1e-5):
                end = i + 1
                break
        engy = engy[start:end]
        f0 = f0[start:end]

        dataLen = len(engy)
        k = float(normalLen - 1) / float(dataLen - 1)

        x = [i * k for i in xrange(dataLen)]
        newX = [i * 1.0 for i in xrange(normalLen)]
        newX[-1] = x[-1]
        # tck = spi.splrep(x, engy)
        # newEngy = spi.splev(newX, tck)
        # tck = spi.splrep(x, f0)
        # newF0 = spi.splev(newX, tck)
        func = spi.interp1d(x, engy, kind='cubic')
        newEngy = func(newX)
        func = spi.interp1d(x, f0, kind='cubic')
        newF0 = func(newX)

        engyfile = open("../data-process-output/trim-interpolation/train/" + label + "/" + dataname + ".engy", "w")
        f0file = open ("../data-process-output/trim-interpolation/train/" + label + "/" + dataname + ".f0", "w")
        for i in xrange(normalLen):
            engyfile.write("%.5f\n" % newEngy[i])
            f0file.write("%.5f\n" % newF0[i])
        engyfile.close()
        f0file.close()