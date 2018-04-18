# -*- coding: utf-8 -*-
"""
Graphtec GL100で記録したデータをGL100-APSを用いて
バイナリから変換した際に生成されるcsvファイルを解析、表示するためのモジュール
"""

import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series

def prepare_dataset(filename):
    """
    csvファイルを読み込んで、扱える形のデータフレームに整形する関数
    秒以上の列とms以下の列が分離しているために、前処理が必要

    Argument
    filename->str:          ファイル名のパス

    return
    Data->pandas.DataFrame: 日時と温度を保持するデータフレーム
    """
    #load dataset
    data = pd.read_csv(filename, skiprows=17, encoding="shift-jis")
    data.columns=(["No", "Time", "ms", "Temp", "1", "A12345678", "A1234", "A1"])

    #reshape dataset
    index = np.arange(0,len(data))
    #data["Time"](yy/MM/DD hh:mm:ss)->hh:mm:ss
    #hh:mm:ss + ms->datetime
    date = [str(data["Time"][i]).split(" ") for i in index]
    date = [date[i][1] + str(":") + str(data["ms"][i]*10**3) for i in index]
    date = [date[i].split(":") for i in index]
    date = [datetime.time(int(date[i][0]), int(date[i][1]), int(date[i][2]), int(date[i][3])) for i in index]

    #make dataset
    Data =DataFrame(np.c_[date, data["Temp"]])
    Data.columns=(["date", "temperature"])
    return Data

def plot_graph(data, start=None, end=None, title=None, figsize_mm=None, filename=None):
    """
    前処理されたデータフレームを画像化する関数

    Argument
    data->DataFrame:            前処理されたデータ

    Keyword argument
    start->int:                 開始点のインデックス
    end->int:                   終了点のインデックス
    title->str:                 plt.title()に与える画像タイトル
    figsize_mm->(float, float): 画像のサイズ指定
    filename->str:              保存ファイル名。与えられたら画像出力する 
    """
    #figsize:[inch]
    if figsize_mm:
        #mm->inch
        figsize = np.array(figsize_mm) / 25.4
        plt.figure(figsize=figsize)
    plt.plot(data["date"][start:end], data["temperature"][start:end])
    plt.xlim(data.date[start:end].min(), data.date[start:end].max())
    plt.grid()
    plt.xlabel("time[hh:mm:ss]")
    plt.ylabel("temperature")
    plt.xticks(rotation=30)
    if title:
        plt.title(title)
    #optimize layout
    plt.tight_layout()

    if filename == None:
        plt.show()
    else:
        filename = r"./" + filename + ".png"
        plt.savefig(filename, dpi=180)


def main():
    filename = r""
    prepare_dataset(filename)

if __name__ == '__main__':
    main()
