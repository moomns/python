#!/c/Python34/python
# coding: utf-8
import os
import datetime

def main():
    #指定したディレクトリ以下に一括処理を施す
    directory = r"./"
    i = 0
    start = datetime.datetime.today()
    print("start ", start)
    for dirpath, dirnames, filenames in os.walk(directory):
        if i==0:
            print(dirpath)
            i = i + 1
        else:
            print(dirpath)
            i = i + 1
    end = datetime.datetime.today()
    print("e n d ", end)
    print("     ", end - start)