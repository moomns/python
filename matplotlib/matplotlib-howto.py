# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


#<matplotlib>
#plotの仕方
#pltで記述する方法

plt.plot(data_x, data_y, label="label", alpha="transparent[%]", "b--")
plt.xlabel("X label")
plt.ylabel("Y label")
plt.xlim([start, end])
plt.ylim([start, end])
plt.title("Title")
plt.grid()
plt.legend()
plt.show()
or
plt.savefig("test.png", dpi=180)

#メソッドで記述する方法
#空の図を定義
fig = plt.figure()
#subplot(n m selector)/n:行 m:列 selector:どの図か
ax1 = fig.add_subplot(111, figsize=(a,b))
ax1.plot(data_x, data_y, label="label", alpha="transparent[%]", "b--")
ax1.set_ylabel("X label")
ax1.set_xlabel("Y label")
ax1.set_xlim([start, end])
ax1.set_ylim([start, end])
ax1.set_title("Title")
ax1.grid()
ax1.legend()
fig.show()
or
fig.savefig("test.png", dpi=180)

#図の背景の透明化
fig = plt.figure()
#背景の透明化
fig.patch.set_alpha(0.)
#imshowにおいて特定の条件を満たす値を描画せず透明のままにする
np.ma.masked_where(data, condition)

#軸の数値配列のget
ax.get_xticks()
#現在の軸の数値情報を利用した、軸ラベルの書き換え
ax.set_xticklabels([x*10**3 for x in ax.get_xticks().tolist()])

#図のサイズ変更
#row, col[inch] 初期設定は(8, 6)
plt.figure(figsize=(row,col))

#サブプロット間を詰める
plt.tight_layout()

#3次元データの2次元表示(色で)
X, Y = np.meshgrid(x,y)
Z = function_vectorized(X, Y)
#Zの値で塗りつぶし
plt.imshow(Z, interpolation="none", extent=[xmin, xmax, ymax, ymin])
plt.gca().invert_yaxis()
or
plt.imshow(Z, origin="lower", interpolation="none", extent=[xmin, xmax, ymin, ymax])
#Zの値で等高線
plt.contour(X,Y,Z)
#Zの値で等高線を書き、その間を塗りつぶす
#zの値の境界値
levels = [0, 2, 3, 10, 56]
plt.contourf(X,Y,Z, levels)

#3D棒グラフ
from mpl_toolkits.mplot3d import Axes3D

d = np.arange(0+1, 16+1)

xx, yy = np.meshgrid(np.arange(0, 4), np.arange(0, 4))
xx = xx.flatten()
yy = yy.flatten()
zz = np.zeros(len(xx))

dx=np.full(len(xx), 0.5)
dy=np.full(len(yy), 0.5)

ax=plt.figure().add_subplot(111, projection='3d')
ax.bar3d(xx, yy, zz, dx, dy, d, color="gray")

ax.set_xlabel("n")
ax.set_ylabel("m")
ax.set_zlabel("dmn", rotation=-90)

ax.set_xticklabels([int(x) if int(x)==x else "" for x in ax.get_xticks().tolist()])
ax.set_yticklabels([int(x) if int(x)==x else "" for x in ax.get_yticks().tolist()])

plt.tight_layout()

#画像保存時に図の余白をできるだけ削除する
plt.tight_layout()
plt.savefig("1331.pdf", dpi=300, bbox_inches = 'tight', pad_inches=0)

#colorbarをグラフにフィットさせる
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.figure()
ax = plt.gca()
#imshowでは左上が原点. pcolormesh(X, Y, Z)では左下が原点
im = ax.imshow(np.arange(100).reshape((10,10)), interpolation="none", extent=[xmin, xmax, ymin, ymax])
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)

#or
divider = make_axes_locatable(ax_im)
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
fig.add_axes(ax_cb)
cb = plt.colorbar(im, cax=ax_cb)
ax_cb.tick_params(labelsize=ticks_size)

#colorbarをグラフにフィットさせる 横向き
divider = make_axes_locatable(ax_im)
ax_cb = divider.new_vertical(size="5%", pad=0.3, pack_start=True)
fig.add_axes(ax_cb)
#colorbarに表示したいticksを指定可能
cb = plt.colorbar(im, cax=ax_cb, orientation="horizontal", ticks=[-0.8, -0.3, 0.3, 0.8])
ax_cb.tick_params(labelsize=ticks_size)
ax_cb.xaxis.set_ticks_position('top')
#colorbarのtickの一部のみを消せる
ax_cb.axes.get_xaxis().set_ticklabels([-0.8, "", "", 0.8])
cb.set_label(label="ΔF/F0 (%)",size=14, labelpad=-28)

#軸の文字サイズ変更
ax1.tick_params(axis='both', which='major', labelsize=12)

#軸反転
ax.invert_yaxis()
ax.invert_xaxis()

#注釈
#xy:注釈を入れる点の座標
#xytext:注釈文の座標
#arrowprops:矢印を使う時のオプション　テキストのみの場合はいらない
plt.annotate("注釈", xy=(x, y), xytext=(x, y), arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))

#対数軸
plt.xscale("log")
plt.yscale("log")

#二軸グラフの書き方
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(data["time[s]"]*10**6, data["Response[V]"]*10**3, label="Response", color="b")
ax1.set_ylabel("Voltage[mV]")
ax1.set_xlabel("time[us]")
ax1.set_ylim(-10, 12)
ax1.grid()
#2軸目を右側につける
ax2 = ax1.twinx()
ax2.plot(data["time[s]"]*10**6, data["Stimulation[V]"], label="Stimulation", color="g", alpha=0.5)
ax2.plot(data["time[s]"]*10**6, data["Trigger[V]"], label="Trigger", color="r", alpha=0.5)
ax2.set_ylabel("Voltage[V]")
ax2.set_ylim(-10, 12)
ax2.set_title("Stim(0-3V->Amp) Average32 Far from the sensor")
ax2.grid()
#凡例をax1, ax2とで共通にする
#これをせずにax1.legend(), ax2.legend()とすると一つのボックス内に描写されない
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc=1)

plt.savefig("ResStimTri.png", dpi=180)

#空の図の定義
fig = plt.figure()
#サブプロットを追加しないと図には書けない
#n×mの図のうち1番目の図を選択する
ax1 = fig.add_subplot(n, m, 1)

#ある値から値までの間を塗りつぶす
plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='training accuracy')
plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy')
plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.0])


#一括して同じ書式の図を作る
fig, axes = plt.subplots(nrows=n, ncols=m, sharex="共通のx軸使用", sharey="共通のy軸使用")

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

#seabornによって変更されたスタイルをリセット
sns.reset_orig()
#同時分布　結合分布
sns.jointplot(data1, data2)
#同時分布　色の濃淡で
sns.jointplot(data1, data2, kind="hex")
#データの分布を見る カーネル密度推定、ヒストグラム、ラグプロット
sns.distplot(dataset, rug=True, hist=True, bins=25,
    kde_kws={"color":"indianred", "label":"KDE PLOT"},
    hist_kws={"color":"blue", "label":"HISTGRAM"})
#箱ひげ図
sns.boxplot(data=[data1,data2], orient="v")
#バイオリンプロット
sns.violinplot(data, inner="stick")
#回帰直線+散布図+信頼区間
sns.lmplot("X", "Y", dataset, order=1)
#離散値について回帰直線+散布図+信頼区間　重なりをずらす
sns.lmplot("size", "tip_pect", tips, x_jitter=0.2, x_estimator=np.mean)
#seabornの描画をmatplotlibのsubplotに載せる

fig, (axis1, axis2) = plt.subplots(1,2,sharey=True)
sns.regplot("total_bill", "tip_pect", tips, ax=axis1)
sns.violinplot(y="tip_pect", x="size", data=tips.sort("size"), ax=axis2)
#ヒートマップ
flight_dframe=sns.load_dataset("flights")
flight_dframe=flight_dframe.pivot("y:month", "x:year", "data:passengers")
sns.heatmap(flight_dframe, annot=True, fmt="d", center=flight_dframe.loc["January", 1955])
#クラスタリング デンドログラム
sns.clustermap(flight_dframe, standard_scale=1, col_cluster=True, z_score=1)
#pairplotの上下を変える
#通常
sns.pairplot(tech_rets.dropna())
#変える
returns_fig=sns.PairGrid(tech_rets.dropna())
returns_fig.map_upper(plt.scatter, color="purple")
returns_fig.map_lower(sns.kdeplot, cmap="cool_d")
returns_fig.map_diag(plt.hist, bins=25)
#カテゴリカルデータの図示
sns.factorplot("Y", data=dataset, hue="kind")

#matplotlibでType1のフォントを埋め込む
#
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True
#plt.rcParams["pdf.fonttype"] = 1
#TrueType->42 OpenType/PostScript->1

#日本語フォントの使用
from pylab import *
import matplotlib.font_manager as fm
prop = fm.FontProperties(fname='N:\\WINDOWS\\Fonts\\ipamp.odf')
xlabel(u'本日は晴天なり', fontproperties=prop)
show()

or

IPAexフォントをダウンロードし、以下に保存
Windows/Fonts、 Python27\Lib\site-packages\matplotlib\mpl-data\fonts\ttf

import matplotlib as mpl
mpl.rcParams['font.family']='IPAexGothic'
