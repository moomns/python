# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from time_series_class import *

def read_oscilloscope_data(filename_input, point=4000):
    """
    オシロスコープMemoryPrime GDS-1000 SeriesのSave Allで保存したcsvファイルを
    扱いやすい形にして読み込む

    Argument
    filename_input:読み込むcsvファイルの名前

    Keyword Argument
    point:csv内に保存されているデータ点数

    Return
    ts:TimeSeries(時系列データを自動でFFTするクラス)に格納した電圧データ
    """

    data = pd.read_csv(filename_input, header=None)
    scale = np.float(data[1][8])
    dt = np.float(data[1][11])
    data = pd.DataFrame(data.ix[16:, 0:1], dtype=np.float64)
    data.index = np.arange(0, len(data))
    data.columns = (["point", "Voltage[V]"])
    #なぜか余分に読み込まれるファイルがあるので、その欠損値を排除
    data = data.dropna()

    fs = int(1./dt)
    ts = TimeSeries(data["Voltage[V]"], fs)
    return ts