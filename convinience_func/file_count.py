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
                index = re.search('.mp3', file) or re.search('.wav', file) or re.search('.m4a', file)# 拡張子がjpgのものを検出
                if index:# jpgの時だけ（今回の場合は）カウンタをカウントアップ
                    count = count + 1
        except FileNotFoundError as e:
            print(e)

print(count)# ファイル数の表示
