import os

dataLabel = ["one", "two", "three", "four"]

dataRoot = "../toneclassifier/train"

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
        os.makedirs("../data-process-output/ignore-low-eneg-frequences/train/" + label)
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
        count = 0
        enegySum = 0.0
        for i in xrange(dataLen):
            if abs(f0[i] - 0) < 1e-5:
                enegySum += engy[i]
                count += 1

        meanEngy = enegySum / count
        for i in xrange(dataLen):
            if engy[i] <= meanEngy:
                f0[i] = 0.0
        engyfile = open("../data-process-output/ignore-low-eneg-frequences/train/" + label + "/" + dataname + ".engy", "w")
        f0file = open ("../data-process-output/ignore-low-eneg-frequences/train/" + label + "/" + dataname + ".f0", "w")
        for i in xrange(dataLen):
            engyfile.write("%.5f\n" % engy[i])
            f0file.write("%.5f\n" % f0[i])
        engyfile.close()
        f0file.close()