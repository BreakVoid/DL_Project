#!/usr/bin/env python

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from math import floor
import scipy.io.wavfile as wav
import numpy as np


for j in xrange(40):
    string_input = raw_input()
    sig = string_input.split()
    for i in xrange(len(sig)):
        sig[i] = int(float(sig[i]))
    sig = np.int_(sig)
    mfcc_feat = mfcc(sig, samplerate=70)
    print j
    print mfcc_feat
    print len(mfcc_feat)
    print len(mfcc_feat[0])
    print ("")


