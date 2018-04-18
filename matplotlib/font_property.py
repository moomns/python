<pre class="prettyprint">#コメントは#マークを使うか"""hogehoge"""とする．
 
#ライブラリをインポート
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#import datetime #ここらへんのパッケージは昔必要になったことがあったようななかったような
#import matplotlib.colors as colors
#import matplotlib.finance as finance
#import matplotlib.dates as mdates
#import matplotlib.ticker as mticker
#import matplotlib.mlab as mlab
#import matplotlib.font_manager as font_manager
 
 
pi = np.pi # 円周率が必要な場合はnumpy（インポート設定からnp.~とする），このように出力できる
e = np.e # 自然対数とかも出力可能
###############################################################################
#                               Graph Setting                                 #
###############################################################################
###############################################################################
mpl.rcParams['font.family'] = 'Arial' #使用するフォント名
mpl.rcParams['pdf.fonttype'] = 42 #このおまじないでフォントが埋め込まれるようになる
params = {'backend': 'ps', # バックエンド設定
          'axes.labelsize': 8, # 軸ラベルのフォントサイズ
          'text.fontsize': 8, # テキストサイズだろう　　　　　　
          'legend.fontsize': 8, # 凡例の文字の大きさ
          'xtick.labelsize': 8, # x軸の数値の文字の大きさ
          'ytick.labelsize': 8, # y軸の数値の文字の大きさ
          'text.usetex': False, # 使用するフォントをtex用（Type1）に変更
          'figure.figsize': [10/2.54, 6/2.54]} # 出力画像のサイズ（インチなので2.54で割る）
mpl.rcParams.update(params)
 
###############################################################################
#                                 Data Input                                  #
###############################################################################
###############################################################################
#データ読み込みにはnumpyのloadtxtを使用すると便利
filename = "sample.txt"
data = np.loadtxt(filename, #ファイルの名前
                  skiprows=1, #この場合ファイルの一行目をスキップする
                  dtype={'names':('data1',
                                  'data2',
                                  'data3'),
                        'formats':('f8',
                                   'f8',
                                   'f8'
                               )})
 
###############################################################################
#                                 Make Graph                                  #
###############################################################################
###############################################################################
plt.rc('axes', grid=True)#
plt.rc('grid', color='0.8', linestyle='-.', linewidth=0.5)#
bs = plt.figure(facecolor='white')
base_ax = plt.axes([0.15,0.15,0.75,0.75])
ax1 = plt.plot(data['data1'], data['data2'], 'ro-', label='Sample1')#散布図の描画．凡例のラベル設定はここで行う
ax2 = plt.plot(data['data1'], data['data3'], 'bs--', label='Sample2')
lx = plt.xlabel(u'x line [\u03bcm]')#x軸のラベル
ly = plt.ylabel(u'y line [\u03bcm]')#y軸のラベル
tx = plt.xticks(np.arange(0, 2.1*pi, 0.5*pi),#np.arange(0, 2*pi, 0.5*pi)とすると2*piが含まれない
                [u'0',u'0.5\u03C0',u'\u03C0',u'1.5\u03C0',u'2\u03C0']) #xラベルの設定
                #$\pi$表記でTex形式出力も可能
limx = plt.xlim(0, 2*pi) # xの範囲
leg = plt.legend(loc='best', shadow=True, fancybox=True)#凡例出力
leg.get_frame().set_alpha(0.5)#凡例を半透明にする
plt.savefig('sample.pdf')# PDF形式で保存する場合，フォントが埋め込まれる
plt.savefig('sample.eps')# EPS形式で保存する場合，フォントはアウトライン化される
plt.savefig('sample.png', dpi=100)# png形式ではdipの調整ができる．デフォルトでは80だったはず
plt.show()#グラフ出力．最後に持ってくる（savefigがうまく動かなかったりするため）
</pre>