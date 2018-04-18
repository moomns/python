# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal

def downsampling_using_decimate(data, fs, resample_interval):
    """
    アンチエイリアシングフィルタをかけた後にダウンサンプリングする
    signal.decimateを用いて、時系列データをダウンサンプリングする

    Argument
    data:時系列データ
    fs:時系列データのサンプリング周波数
    resample_interval:何点おきにリサンプリングするか int

    Return
    downsampled_data:ダウンサンプリング後の時系列データ
    downsampled_fs:ダウンサンプリング後のサンプリング周波数
    """

    downsampled_fs = fs // resample_interval
    downsampled_data = signal.decimate(data, resample_interval, zero_phase=True)

    return downsampled_data, downsampled_fs
