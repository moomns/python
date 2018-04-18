#!/c/Python34/python
# coding: utf-8

import numpy as np
import scipy.signal

class TimeSeries(object):
    """
    時系列データを扱ってフーリエ変換を施すクラス定義 
    http://nbviewer.ipython.org/gist/mkatsura/6052138 を改変

    Argument
    data:1次元時系列データ配列
    fs:サンプリング周波数[Hz]

    Member
    data:1次元時系列データ配列
    fs:サンプリング周波数[Hz]
    nyq:ナイキスト周波数[hz]
    dt:サンプリング時間[s]
    N:データ点数
    DeltaF:周波数分解能[Hz]
    time:dataに関する時系列[s]
    frequency:FFT後の周波数目盛り[Hz]

    fft_amp:FFT振幅スペクトル強度
    fft_phase:FFT位相スペクトル強度

    b_FIR
    h_FIR
    freqList_FIR
    FIRed_data

    b_IIR
    h_IIR
    freqList_IIR
    IIRed_data

    cumsum

    """

    def __init__(self, data, fs):
        self.data = data
        self.fs = fs
        self.nyq = self.fs / 2
        self.dt = 1. / self.fs
        self.N = len(data)
        self.DeltaF = fs / self.N
        #明示的に型変換しないと将来的にエラーとなる
        self.NyN = int(np.floor(0.5 * self.N) + 1)
        self.fft()
        self.time = np.arange(self.N) / fs
        self.frequency = np.arange(self.NyN) * self.DeltaF
        self.power_func()

    def power_func(self):
        power = (self.fft_amp ** 2) * 0.5
        power[0] = power[0] * 2.0  #直流に関しては0.5倍は不要なので、戻す。
        self.power = power

    def fft(self):
        data_fft = np.fft.fft(self.data)
        self.fft_amp = np.zeros(self.NyN)
        self.fft_phase = np.zeros(self.NyN)
        N = self.N
        for i, _data_fft in enumerate(data_fft[:self.NyN]):
            if i == 0:
                self.fft_amp[i] = np.real(_data_fft) / N
                self.fft_phase[i] = 0.0
            elif i == self.N * 0.5:
                self.fft_amp[i] = np.absolute(_data_fft) * (1.0 / N)
                self.fft_phase[i] = 0
            else:
                self.fft_amp[i] = np.absolute(_data_fft) * (2.0 / N)
                self.fft_phase[i] = np.angle(_data_fft) * 180.0 / np.pi

        _exp = self.fft_amp * np.exp(-1.0j * self.fft_phase * np.pi / 180.0)
        self.fft_cosine = np.real(_exp)
        self.fft_sine = np.imag(_exp)
        return self.fft_amp, self.fft_phase

    def firfilter(self, fc=100*10**3, numtaps=255, filter_option="lpf"):
        fc = fc / self.nyq
        if filter_option == "lpf" :
            self.b_FIR = scipy.signal.firwin(numtaps, fc, window="hamming")
        elif filter_option == "hpf" :
            self.b_FIR = scipy.signal.firwin(numtaps, fc, pass_zero=False, window="hamming")

        w, h = scipy.signal.freqz(self.b_FIR, 1)
        self.freqList_FIR = []
        self.h_FIR = h
        for i in w:
            self.freqList_FIR.append(i * (self.fs / 2.0) / np.pi)

        self.FIRed_data = scipy.signal.lfilter(self.b_FIR, 1, self.data)

    def iirfilter(self, fc=100*10**3, numtaps=3, btype="lowpass"):
        fc = fc / self.nyq
        b, a = scipy.signal.iirfilter(N=numtaps, Wn=fc, btype=btype, 
            analog=False, ftype="butter", output="ba")
        self.b_IIR = b
        w, h = scipy.signal.freqz(b, a)
        self.h_IIR = h
        self.freqList_IIR = []
        for i in w:
            self.freqList_IIR.append(i * (self.fs / 2.0) / np.pi)

        self.IIRed_data = scipy.signal.lfilter(b, a, self.data)

    #dataにはno stim配列や平均値を入れることで0中心に補正
    def integrate(self, data = 0, N=100, r=0.7*10**(-3)):
        tmp = (self.filtered_data - data) * self.dt
        S = np.pi * r ** 2
        self.cumsum = (tmp / (N * S)).cumsum()

