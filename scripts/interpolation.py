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
        os.makedirs("../data-process-output/interpolation/train/" + label)
    except OSError as err:
        pass

    for dataname in dataset:
        engyfile = open(subsetPath + "/" + dataname + ".engy", "r")
        f0file = open(subsetPath + "/" + dataname + ".f0", "r")
        engy = map(float, engyfile.readlines())
        f0 = map(float, f0file.readlines())
        engyfile.close()
        f0file.close()

        dataLen = len(engy)
        k = float(normalLen) / float(dataLen)

        x = [i * k for i in xrange(dataLen)]
        newX = [i for i in xrange(normalLen)]
        tck = spi.splrep(x, engy)
        newEngy = spi.splev(newX, tck)
        tck = spi.splrep(x, f0)
        newF0 = spi.splev(newX, tck)


        engyfile = open("../data-process-output/interpolation/train/" + label + "/" + dataname + ".engy", "w")
        f0file = open ("../data-process-output/interpolation/train/" + label + "/" + dataname + ".f0", "w")
        for i in xrange(normalLen):
            engyfile.write("%.5f\n" % newEngy[i])
            f0file.write("%.5f\n" % newF0[i])
        engyfile.close()
        f0file.close()