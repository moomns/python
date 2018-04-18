# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


# assumed sample rate of OpenBCI
fs_Hz = 250.0

# create the 60 Hz filter
bp_stop_Hz = np.array([59.0, 61.0])
b, a = signal.butter(2,bp_stop_Hz/(fs_Hz / 2.0), 'bandstop')
# create the 50 Hz filter
bp2_stop_Hz = np.array([49, 51.0])
b2, a2 = signal.butter(2,bp2_stop_Hz/(fs_Hz / 2.0), 'bandstop')


# compute the frequency response
w, h = signal.freqz(b,a,1000)
w, h2 = signal.freqz(b2,a2,1000)
f = w * fs_Hz / (2*np.pi)             # convert from rad/sample to Hz

