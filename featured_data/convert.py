#!/usr/bin/env python

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from math import floor
import scipy.io.wavfile as wav
import numpy as np


import os

f0_file = open("../data/test_f0s", "r")

# for j in xrange(40):
#     string_input = raw_input()
#     sig = string_input.split()
#     for i in xrange(len(sig)):
#         sig[i] = int(float(sig[i]))
#     sig = np.int_(sig)
#     mfcc_feat = mfcc(sig, samplerate=70)
#     print j
#     print mfcc_feat
#     print len(mfcc_feat)
#     print len(mfcc_feat[0])
#     print ("")

f0s = []

for line in f0_file.readlines():
    f0s.append(map(int, map(float, line.split())))

labels = []
for line in open("../data/test_labels", "r").readlines():
	labels.append(int(line))

i = 0
for f0 in f0s:
	sig = np.int_(f0)
	mfcc_feat = mfcc(sig, samplerate=70)
	out_file = open("featured_test_f0s", "a")
	for j in xrange(len(mfcc_feat[0])):
		for k in xrange(len(mfcc_feat)):
		    out_file.write("%.8f " % mfcc_feat[k][j])
		    out_file.write("\n")

	out_file.close()

