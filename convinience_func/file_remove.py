# 必要なライブラリのインクルード
import os
import re

dir = os.getcwd()# カレントディレクトリのパスを取得

count = 0# カウンタの初期化
for dirpath, dirnames, filenames in os.walk(dir):
    for dirname in dirnames:
        try:
            files = os.listdir(dirpath + r"\\" + dirname)# ファイルのリストを取得
            for file in files:# ファイルの数だけループ
                if '(2).mp3'in file:# jpgの時だけ（今回の場合は）カウンタをカウントアップ
                    os.remove(dirpath + r"\\" + dirname + r"\\" + file)
                    count += 1
        except PermissionError as e:
            print(e)

print(count, "files were removed")# ファイル数の表示
