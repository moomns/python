# -*- coding: utf-8 -*-

import os
import datetime

def batch_processing(func, directory=r"./"):
    """
    指定したディレクトリ以下に一括処理を施す

    Argument
    func:任意ディレクトリ以下に一括処理を施したい関数

    Keyword Argument
    directory:指定が無ければ、実行フォルダ以下を対象とする
    """
    
    i = 0
    start = datetime.datetime.today()
    print("start ", start)
    for dirpath, dirnames, filenames in os.walk(directory):
        if i==0:
            func(dirpath, dirnames, filenames)
            i = i + 1
        else:
            func(dirpath, dirnames, filenames)
            i = i + 1
    end = datetime.datetime.today()
    print("e n d ", end)
    print("     ", end - start)