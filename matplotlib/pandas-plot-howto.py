# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series
import statsmodels.api as sm
from statsmodels.formula.api as glm
from sympy import *

#<pandas>
#Excelで作成したcsvファイルの読み込み
#エンコードをshift-jisに指定しないとDecodeErrorで読み込めない
data = pd.read_csv(filename, encoding="shift-jis")
#読み込み行数指定
pd.read_csv(filename, nrows=2)

#データフレームの定義
data = DataFrame(np.arange(1,10).reshape(3,3), columns=["A", "B", "C"], index=None)

#行へのアクセス
data.ix[0]

#列へのアクセス
data["A"]
data.A
data.ix[:, 0]
data.ix[;, "A"]

#行と列を同時に選択
data.ix[0:1, 0:2]

#条件に合致するものを選択
data[data["A"]>5]

#再インデックス付け
data.reindex(index, columns, method)

#関数の適用
data.apply(function)

#型変換
data_Br.columns = data_Br.columns.astype(np.float64)

#DataFrame, Seriesの要約統計量メソッド
#NAでない要素数
count
#各種要約統計量表示
describe
#データのパーセント点を0～1で求める→四分位数など
quantile
#累積合計値
cumsum
#一次の階差
diff
#相関
corr
#共分散
cov
#特定の行や列と別のシリーズやデータフレームとの相関
corrwith

#csv書き出し
data.to_csv(filename, delimiter="分割用文字", )

#連続データの離散化→ヒストグラムなど
bins=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
cats = pd.cut(data, bins)

#グループング
#グループ化して、それに任意の処理を加える
data.groupby().apply()

#グループングしたオブジェクトから，keyやvalueのみを取り出す
grouped_MaxResponse_f = data["Max Response (%)"][data["amplitude (V)"]>0].groupby(data["amplitude (V)"])
grouped_MaxResponse_f.groups
->
{0.5: [1, 12, 23, 34, 45, 56, 67],
 1.0: [2, 13, 24, 35, 46, 57, 68],
 1.5: [3, 14, 25, 36, 47, 58, 69],
 2.0: [4, 15, 26, 37, 48, 59, 70],
 2.5: [5, 16, 27, 38, 49, 60, 71],
 3.0: [6, 17, 28, 39, 50, 61, 72],
 3.5: [7, 18, 29, 40, 51, 62, 73],
 4.0: [8, 19, 30, 41, 52, 63, 74],
 4.5: [9, 20, 31, 42, 53, 64, 75],
 5.0: [10, 21, 32, 43, 54, 65, 76]}

grouped_MaxResponse_f.groups.keys()
->
dict_keys([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 3.5, 1.5, 4.5, 2.5])

grouped_MaxResponse_f.groups.values()
->

#列の値をもとに演算を施し、その演算結果を列として追加する->applyを用いる
AAFremoved["p_peak_appeared_repaired"] = AAFremoved["p_remark"].apply(lambda data: isinstance(data, str)) | AAFremoved["p_peak_appeared"]
AAFremoved["Max Response (%)"] = AAFremoved["p_max"].apply(lambda F: F if F>0 else 0)

#ピボットテーブル
data.pivot_table()

#クロス集計
data.crosstab()

#縦に連結
pd.concat([A,B])
A.append(B)

#横に連結
pd.concat([A, B], axis=1)
pd.merge(left, right, left_index=True, right_index=True)
A.join(B)

#結合
pd.merge(left, right, on='key', how="outer")

#一般化線形モデル
df = pd.read_csv(filename)
#最初が説明変数と応答変数の関係、familyがyが従うものとして仮定する分布
mdl = glm("y~x", data=df, family=sm.families.Poisson(sm.families.links.log))
fitting = mdl.fit()
print(fitting.summary())

#<matplotlib>
#plotの仕方
#空の図の定義
fig = plt.figure()
#サブプロットを追加しないと図には書けない
#n×mの図のうち1番目の図を選択する
ax1 = fig.add_subplot(n, m, 1)


#一括して同じ書式の図を作る
fig, axes = plt.subplots(nrows=n, ncols=m, sharex="共通のx軸使用", sharey="共通のy軸使用")

#図の保存
fig.savefig()

#グラフの描画
data.plot(kind="グラフの種類")
#kind
#棒グラフ
bar
#折れ線グラフ
デフォルト
#ヒストグラム
hist
#散布図
scatter
#散布図行列
scatter_matrix
#画像化
#option=interpolationオプションについて
#None  ぼかしがかかる
#none, 'nearest' ぼかしがかからない
#cmap：カラーマップ
plt.imshow(data[No], cmap=plt.get_cmap("gray"), interpolation=option)
#色の意味、カラーバー
plt.colorbar()

#<Sympy>
#使う変数を定義
x,y,z,t = symbols("x y z t")
#関数を定義
f = x ** 2 - x - 6
#xについて不定積分
integrate(f, x)
#xについて定積分
integrate(f, (x, a, b))
#xについて微分

#<Scipy>
#func(x)を(a,b)で数値積分
scipy.integrate.quad(func, a, b)

#データを取り除く　元のデータフレームに適用するならinplace=True
DataFrame.drop()
#