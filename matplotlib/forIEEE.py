
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')


# In[2]:



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.rcParams['font.family'] = 'Arial'
from scipy import ndimage
from scipy.stats import scoreatpercentile
import os
import cv2
import datetime
from tqdm import tqdm

class BVData(object):
    def __init__(self, background, raw, analog=None, width=188, height=160, n_frame=80, n_average=20):
        self.background = background
        self.raw = raw
        self.analog = analog
        self.width = width
        self.height = height
        self.n_frame = n_frame
        self.n_average = n_average
        self.sampling_rate = 50*10**(-3)
        self.time = np.arange(0,self.n_frame) * self.sampling_rate
        self.frame_onset = 8

    def pre_processing(self, fileROI=None):
        #ROIの抽出
        if fileROI:
            self.processed = np.zeros((self.n_frame) * self.height * self.width).reshape(
                self.n_frame, self.height, self.width)
            self._extract_ROI(fileROI)
        else:
            self.processed = np.array(self.raw).astype(np.float64)

        self._spartial_filter()
        self._temporal_filter()

        #加算されただけのデータをAverage回数で割る
        self.processed = self.processed / self.n_average

        #基準フレームを設定　刺激がframe8で提示されるため，
        #基本はframe0-8の平均をrefに
        self._make_reference_frame()

        #refを基準としたdeltaF/F0行列を計算
        self._calculate_fulorescence_change_from_ref()

    def analyze_parameter(self):
        ##################
        # 正のピーク解析 #
        ##################
        #正の最大蛍光変化を示した時の時間と座標
        #argmaxでは1次元に直した時の座標が返ってくる
        self.results = {}
        self.results["p_frame"], self.results["p_height"], self.results["p_width"]             = np.unravel_index(self.response.argmax(), self.response.shape)
        #刺激前に最大を取っていないかの確認
        if self.results["p_frame"] < self.frame_onset:
            self.results["p_alert"] = True
        else:
            self.results["p_alert"] = False
        #最大蛍光変化率
        self.results["p_max"] = self.response[self.results["p_frame"],
         self.results["p_height"], self.results["p_width"]]
        #反応有無の判定
        if self.results["p_alert"] == False and self.results["p_max"] >= 0.2:
            self.results["is_positive"] = True
        else:
            self.results["is_positive"] = False
        #peak latencyの計算
        self.results["p_peak_latency"] = (self.results["p_frame"]-self.frame_onset)*self.sampling_rate
        #half latencyの計算
        self.results["p_half_latency"] = np.argmax(self.response
            [:, self.results["p_height"], self.results["p_width"]]>=self.results["p_max"]/2)
        self.results["p_half_latency"] = (self.results["p_half_latency"]
            -self.frame_onset)*self.sampling_rate
        #durationの計算
        p_duration = np.where(self.response[:, self.results["p_height"], self.results["p_width"]]
            >=self.results["p_max"]/2)
        self.results["p_duration"] = (p_duration[0][-1] - p_duration[0][0]) * self.sampling_rate
        #最大蛍光変化を示した際に，最大蛍光変化の50%,75%以上の値を示すピクセル数
        self.results["p_S50"] = len(np.where(self.response[
            self.results["p_frame"]]>=self.results["p_max"]/2)[0])
        self.results["p_S75"] = len(np.where(self.response[
            self.results["p_frame"]]>=self.results["p_max"]/4*3)[0])

        ##################
        # 負のピーク解析 #
        ##################
        #負の最大蛍光変化を示した時の時間と座標
        #argmaxでは1次元に直した時の座標が返ってくる
        self.results["n_frame"], self.results["n_height"], self.results["n_width"]             = np.unravel_index(self.response.argmin(), self.response.shape)
        #刺激前に最大を取っていないかの確認
        if self.results["n_frame"] < self.frame_onset:
            self.results["n_alert"] = True
        else:
            self.results["n_alert"] = False
        #最大蛍光変化率
        self.results["n_max"] = self.response[self.results["n_frame"],
         self.results["n_height"], self.results["n_width"]]
        #反応有無の判定
        if self.results["n_alert"] == False and self.results["n_max"] <= -0.2:
            self.results["is_negative"] = True
        else:
            self.results["is_negative"] = False
        #peak latencyの計算
        self.results["n_peak_latency"] = (self.results["n_frame"]-self.frame_onset)*self.sampling_rate
        #half latencyの計算
        #boolの配列を渡すと，argmaxは初めにTrueになるindexを返す
        self.results["n_half_latency"] = np.argmax(self.response
            [:, self.results["n_height"], self.results["n_width"]]<=self.results["n_max"]/2)
        self.results["n_half_latency"] = (self.results["n_half_latency"]
            -self.frame_onset)*self.sampling_rate
        #durationの計算
        n_duration = np.where(self.response[:, self.results["n_height"], self.results["n_width"]]
            <=self.results["n_max"]/2)
        self.results["n_duration"] = (n_duration[0][-1] - n_duration[0][0]) * self.sampling_rate
        #最大蛍光変化を示した際に，最大蛍光変化の50%,75%以上の値を示すピクセル数
        self.results["n_S50"] = len(np.where(self.response[
            self.results["n_frame"]]<=self.results["n_max"]/2)[0])
        self.results["n_S75"] = len(np.where(self.response[
            self.results["n_frame"]]<=self.results["n_max"]/4*3)[0])

    def to_csv(self):
        #背景データと差分データを結合して，csvを吐き出す
        #BV_anaのcsv吐き出しとは異なり，
        #各フレーム間に改行が含まれない
        pd.DataFrame(np.r_[self.background, 
            self.raw.reshape(self.n_frame * self.height, self.width)]).to_csv(
            "data_raw.csv", index=None, header=None)
        pd.DataFrame(np.r_[self.background, 
            self.processed.reshape(self.n_frame * self.height, self.width)]).to_csv(
            "data_processed.csv", index=None, header=None)

    def draw_image_selected_frame(self, frame, threshold_up=0.3, threshold_down=-0.3, 
        gain=15, offset=-192, percentile_min=1, percentile_max=90, gamma=1.0, 
        color=cm.seismic, color_vmin=None, color_vmax=None, format="png", dpi=300):
        #threshold_up:0-threshold_up%までの蛍光変化を示すピクセル点を描画しない
        #threshold_down:threshold_down-0%までの蛍光変化を示すピクセル点を描画しない
        #gain, offset: 背景画像輝度を線形変換 BV_anaではgain=15くらいoffset=-192
        #percentile_min,max:背景画像輝度の表示範囲は百分位数で切る 1-90%くらいでBV_ana標準に近い
        #gamma:背景画像に対してかけるgamma補正パラメータ gamma=1なら補正しない Maxは3くらい？
        #color:蛍光変化のカラーテーブル
        #color_vmin, vmax:蛍光変化のカラースケールの下限、上限値 -vmin=vmaxとなるように指定する
        #format:保存の際のフォーマット eps,ps保存は環境依存で正常に行えない場合がある 原因不明
        #dpi:解像度 通常は300で高品質とされる 論文などでは1600ぐらいを要求される場合あり

        #BV_anaのように背景データを線形変換する
        #background dataはint16なので演算過程でオーバーフローする->int64に前もって変換
        streched = gain * self.background.astype(np.int64) / self.n_average + offset

        #規格化した背景データを描画
        normed = streched /streched.max()
        #gamma補正を背景に施す
        gammaed = 255 * (normed * 255 /255) ** (1 / gamma)

        plt.imshow(gammaed, vmin=scoreatpercentile(gammaed, percentile_min), 
            vmax=scoreatpercentile(gammaed, percentile_max), cmap=cm.gray, interpolation="none")

        #蛍光強度変化データに関して閾値以下の変化を示す蛍光強度変化配列をマスク
        up = self.response[frame] < threshold_up
        down = self.response[frame] > threshold_down
        condition = up & down
        masked = np.ma.masked_where(condition, self.response[frame])
        #カラーマップをBVanaに合わせるなら、
        #import BVcolormap ->cmap=BVcolormap.bv_color
        #と指定する
        plt.imshow(masked,cmap=color,vmin=color_vmin, vmax=color_vmax, interpolation="none")
        plt.xticks(visible=False)
        plt.yticks(visible=False)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig("color_image_"+str(frame)+"."+format, dpi=dpi)

    def draw_normed_image_selected_frame(self, frame, threshold_up=0.3, threshold_down=-0.3, 
        gamma=1.0, color=cm.seismic, color_vmin=None, color_vmax=None):
        #指定されたフレーム内の最大蛍光強度変化で規格化したマップを表示
        #DF/Fmaxを反映できていないため使用不可
        #
        #規格化した背景データを描画
        #暗い場合にはblight_correctionで補正?
        normed_background = self.background /self.background.max()
        gammaed_background = 255 * (normed_background * 255 /255) ** (1 / gamma)
        plt.imshow(gammaed_background, vmin=normed_background.min(), 
            cmap=cm.gray, interpolation="none")

        #規格化した蛍光強度変化データを描画
        normed = self.response[frame] / self.response[frame].max()
        #規格化したデータに関して閾値以下の絶対値変化を示す蛍光強度変化配列をマスク
        up = normed < threshold_up
        down = normed > threshold_down
        condition = up & down
        masked = np.ma.masked_where(condition, normed)
        #カラーマップをBVanaに合わせるなら、
        #import BVcolormap ->cmap=BVcolormap.bv_color
        #と指定する
        plt.imshow(masked,cmap=color,vmin=color_vmin, vmax=color_vmax, interpolation="none")
        plt.xticks(visible=False)
        plt.yticks(visible=False)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig("color_image_"+str(frame)+".png", dpi=300)

    def draw_wave_selected_location(self, height, width):
        fig = plt.figure(figsize=(8,6))
        ax1 = fig.add_subplot(211)
        ax1.plot(np.linspace(0,4,80*20), self.analog[1], label="Stim")
        plt.setp(ax1.get_yticklabels(), visible=False)
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.legend()
        ax2 = fig.add_subplot(212, sharex=ax1)
        plt.axhline(y=0, color="white", linewidth=3)
        ax2.plot(self.time, self.response[:,height, width], 
            label="("+str(height)+", "+str(width)+")")
        ax2.set_ylabel("(%)")
        ax2.legend()
        plt.tight_layout()
        plt.savefig("wave_image_"+str(height)+"_"+str(width)+".png", dpi=300)

    def _extract_ROI(self, filename):
        #ROIの抽出関数
        self._make_mask(filename)
        for index, tmp in enumerate(self.raw):
            self.processed[index] = tmp * self.mask

    def _make_mask(self, filename):
        #マスク配列の作成関数
        self.mask = np.zeros((self.height, self.width),dtype = 'uint8')
        #BV anaで生成したマスク用座標群の読み込み
        polygon = pd.read_csv(filename)
        c = np.array(polygon.ix[:,0])
        r = np.array(polygon.ix[:,1])
        rc = np.array((c,r)).T
        #マスク配列の生成
        cv2.drawContours(self.mask,[rc],0,255,-1)
        self.mask = cv2.cvtColor(self.mask,cv2.COLOR_GRAY2BGR)
        #ROI外が黒255で潰されているため正規化
        self.mask = self.mask[:,:,0]/255

    def _spartial_filter(self, ndim=5):
        #端点の処理がBV_anaと違う
        for index, tmp in enumerate(self.processed):
            #与える配列のデータ型をfloatに変換しないと小数点以下は切り捨てられる
            self.processed[index] = ndimage.uniform_filter(tmp, ndim, mode="reflect")

    def _temporal_filter(self, ndim=5):
        #端点の処理もBV_anaと同じ
        #height*widthの座標の組を生成
        y=np.arange(0,self.height)
        x=np.arange(0,self.width)
        xx, yy = np.meshgrid(x, y)

        for height, width in zip(yy.ravel(), xx.ravel()):
            #背景データに時間フィルタはかけない
            self.processed[:,height, width] = ndimage.uniform_filter(
                self.processed[:, height, width], ndim, mode="nearest")

    def _make_reference_frame(self, start=0, end=8):
        self.ref = self.processed[start:end+1].mean(axis=0)

    def _calculate_fulorescence_change_from_ref(self):
        #刺激前の蛍光量refからの蛍光変化量deltaFを規格化して算出
        self.response = np.zeros((self.n_frame) * self.height * self.width).reshape(
                self.n_frame, self.height, self.width)
        #gsdやcsvに記録される差分データはinverse化されていない
        #普段見ているフラビンの蛍光変化に合わせるため，分母のrefには(-1)を掛ける
        self.response = (self.processed-self.ref)/(self.background+(-1)*self.ref)*100


def read_BV_csv(filename, width=188, height=160, n_frame=80, n_average=20):
    raw = pd.read_csv(filename, header=None).ix[:,:width-1]
    raw = np.array(raw)
    reshaped = raw.reshape(n_frame+1, height, width)
    return BVData(reshaped[0], reshaped[1:], width=width, height=height, n_frame=n_frame, n_average=n_average)

def read_BV_gsd(filename, width=188, height=160, n_frame=80, n_average=20):
    ####################################
    # バイナリファイルの読み出し数設定 #
    ####################################
    #header情報
    header = ("header", "<256c")
    #DataFormat情報
    #data_format = ("data_format","<36h")
    data_format = ("data_format", 
                   [("nDataXsize", "<h"),
                    ("nDataYsize", "<h"),
                    ("nLeftSkip", "<h"),
                    ("nTopSkip", "<h"),
                    ("nImgXsize", "<h"),
                    ("nImgYsize", "<h"),
                    ("nFramesize", "<h"),
                    ("nOrgImgXsize", "<h"),
                    ("nOrgImgYsize", "<h"),
                    ("nOrgFrmsize", "<h"),
                    ("nShift", "<h"),
                    ("nDummy", "<h"),
                    ("dAverage", "<f"),
                    ("dSampleTime", "<f"),
                    ("dOrgSampleTime", "<f"),
                    ("dDummy", "<f"),
                    ("chDum32", "<32c")])
    #外部信号用情報
    ana_signal = ("ana_signal", 
                  [("nChanum", "<h"),
                   ("nRate", "<h"),
                   ("nOfset", "<h"),
                   ("nChNext", "<h"),
                   ("nTimeNext", "<h"),
                   ("nFrameSize", "<h"),
                   ("nShift", "<h"),
                   ("nDummy1", "<h"),
                   ("nDummy2", "<h"),
                   ("nDummy3", "<h")])
    #コントロール用情報
    control = ("control", "<312h")
    
    #背景データ->reshape(height, width)
    background = ("background", "<"+str(height*width)+"h")
    #差分データ全フレーム->reshape(n_frame, height, width)
    raw_data = ("raw_data", "<"+str(n_frame*height*width)+"h")
    #アナログ入力信号情報
    #チャンネル数×フレーム数×時間分解能
    ana_input = ("ana_input", "<"+str(2*n_frame*20)+"h")
    data_type = np.dtype([header, data_format, ana_signal, control, 
        background, raw_data, ana_input])

    ##########################
    # バイナリデータ読み出し #
    ##########################
    with open(filename, "r") as fd:
        chunk = np.fromfile(fd, dtype=data_type, count=1)

    background = chunk[0]["background"].reshape(height, width)
    reshaped = chunk[0]["raw_data"].reshape(n_frame, height, width)
    analog = chunk[0]["ana_input"].reshape(2,n_frame*20)
    return BVData(background, reshaped, analog, width=width, height=height, n_frame=n_frame, n_average=n_average)


# In[3]:

data = read_BV_gsd("1331-Multi500A8.gsd", n_average=10)
ROI = "1331-Multi500A9-Remove-Polygon.csv"
data.pre_processing(fileROI=ROI)
data.analyze_parameter()


# In[4]:

sound2 = read_BV_gsd("1216-4kHz.gsd", n_average=10)
sound2.pre_processing(fileROI="1218-8kHz-Remove-Polygon.csv")
sound2.analyze_parameter()


# In[5]:

TTX = read_BV_gsd("1537-Multi500A8.gsd", n_average=10)
ROI2 = "1414-Multi500A3-Remove-Polygon.csv"
TTX.pre_processing(fileROI=ROI2)
TTX.analyze_parameter()


# In[6]:

import BVcolormap as color


# #2016年に使用していたパラメータ
# ##実際の波形表示ピクセルと若干のずれがあった

# In[37]:

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

dt=0.05

ticks_size=22

frame=20
threshold_up=0.2
threshold_down=-0.2
color_vmin=-0.7
color_vmax=0.7
colormap=color.bv_color_non_gray

height_margin=40
width_margin=20

height1=89
width1=107

height2=119
width2=88

height3=113
width3=33

now = round(dt*frame-0.5, 2)
disp_time=str(now)
h_disp = 150+5
w_disp = 140-width_margin

fig = plt.figure(figsize=(15,8))
#背景の透明化
fig.patch.set_alpha(0.)
gs = gridspec.GridSpec(5,2)

###fig_left
ax_im = fig.add_subplot(gs[:,0])
#BV_anaのように背景データを線形変換する
#background dataはint16なので演算過程でオーバーフローする->int64に前もって変換
streched = data.background.astype(np.int64) / data.n_average
#規格化した背景データを描画
normed = streched /streched.max()
#gamma補正を背景に施す
gammaed = normed
ax_im.imshow(gammaed[height_margin:,:-width_margin], vmin=scoreatpercentile(gammaed, 1), vmax=scoreatpercentile(gammaed, 90), cmap=cm.gray, interpolation="none")

#蛍光強度変化データに関して閾値以下の変化を示す蛍光強度変化配列をマスク
up = data.response[frame] < threshold_up
down = data.response[frame] > threshold_down
condition = up & down
masked = np.ma.masked_where(condition, data.response[frame])
#カラーマップをBVanaに合わせるなら、
#import BVcolormap ->cmap=BVcolormap.bv_color
#と指定する
im = ax_im.imshow(masked[height_margin:,:-width_margin],cmap=colormap,vmin=color_vmin, vmax=color_vmax, interpolation="none", alpha=0.75)

#位置確認
place = np.ones((160, 188))
place[[[height1, height2, height3], [width1, width2, width3]]] = 0
place_mask = np.ma.masked_where(place, place)
ax_im.imshow(place_mask[height_margin1:-height_margin2,:-width_margin], cmap="gray",interpolation="none")

plt.text(width1-4, height1+4-height_margin, "×", color="#55A868", fontsize=48)
plt.text(width2-4, height2+4-height_margin, "×", color="#4C72B0", fontsize=48)
plt.text(width3-4, height3+4-height_margin, "×", color="#C44E52", fontsize=48)
plt.text(width1+5, height1-height_margin, "A", color="#55A868", fontsize=32, weight="bold")
plt.text(width2+5, height2-height_margin, "B", color="#4C72B0", fontsize=32, weight="bold")
plt.text(width3+5, height3-height_margin, "C", color="#C44E52", fontsize=32, weight="bold")
plt.text(w_disp, h_disp-height_margin, disp_time, color="white", fontsize=32)
plt.text(w_disp+30, h_disp-height_margin, "s", color="white", fontsize=32)
plt.xticks(visible=False)
plt.yticks(visible=False)
plt.gca().set_axis_off()

divider = make_axes_locatable(ax_im)
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
fig.add_axes(ax_cb)
#plt.colorbar(im, cax=ax_cb).set_label(label="ΔF/F0 (%)",size=18)
plt.colorbar(im, cax=ax_cb)
ax_cb.tick_params(labelsize=ticks_size)
plt.setp(plt.getp(ax_cb, 'yticklabels'), color='#1A1A1A')

###fig_right_1
ax=fig.add_subplot(gs[1:4,1:])
ax.axhspan(threshold_down, threshold_up, facecolor="gray", alpha=0.3)
ax.axhline(y=0, color="white", linewidth=3)
ax.axvline(x=0, color="white", linewidth=3)
ax.axvline(x=now, linestyle="--", color="black", linewidth=2)
ax.plot(data.time-0.5, data.response[:,height1, width1], linewidth=3, label="site A", color="#55A868")
ax.plot(data.time-0.5, data.response[:,height2, width2], linewidth=3, label="site B", color="#4C72B0")
ax.plot(data.time-0.5, data.response[:,height3, width3], linewidth=3, label="site C", color="#C44E52")
ax.legend(fontsize=ticks_size, loc=4)
ax.set_xlabel("Time (s)", fontsize=26, color="#1A1A1A")
plt.xticks(fontsize=ticks_size, color="#1A1A1A")
#ax.set_ylabel("$\Delta$$F/$$F_0$", fontsize=18)
ax.set_ylabel("ΔF/F0 (%)", fontsize=26, color="#1A1A1A")
plt.yticks(fontsize=ticks_size, color="#1A1A1A")
ax.set_ylim(color_vmin, color_vmax)

gs.tight_layout(fig)
#gs.update(hspace=0.3)
#plt.savefig("1331-500_frame20.pdf", dpi=600)


# In[36]:

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

dt=0.05

frame=20
threshold_up=0.2
threshold_down=-0.2
color_vmin=-0.7
color_vmax=0.7
colormap=color.bv_color_non_gray

height_margin1=20
height_margin2=20
width_margin=20

height1=89
width1=107

height2=119
width2=88

height3=108
width3=35

now = round(dt*frame-0.5, 2)
disp_time=str(now)
h_disp = 150+5
w_disp = 140-width_margin

fig = plt.figure(figsize=(15,8))
#背景の透明化
fig.patch.set_alpha(0.)
gs = gridspec.GridSpec(5,2)

###fig_left
ax_im = fig.add_subplot(gs[:,0])
#BV_anaのように背景データを線形変換する
#background dataはint16なので演算過程でオーバーフローする->int64に前もって変換
streched = sound2.background.astype(np.int64) / sound2.n_average
#規格化した背景データを描画
normed = streched /streched.max()
#gamma補正を背景に施す
gammaed = normed
ax_im.imshow(gammaed[height_margin1:-height_margin2,:-width_margin], vmin=scoreatpercentile(gammaed, 1), vmax=scoreatpercentile(gammaed, 90), cmap=cm.gray, interpolation="none")

#蛍光強度変化データに関して閾値以下の変化を示す蛍光強度変化配列をマスク
up = sound2.response[frame] < threshold_up
down = sound2.response[frame] > threshold_down
condition = up & down
masked = np.ma.masked_where(condition, sound2.response[frame])
#カラーマップをBVanaに合わせるなら、
#import BVcolormap ->cmap=BVcolormap.bv_color
#と指定する
im = ax_im.imshow(masked[height_margin1:-height_margin2,:-width_margin],cmap=colormap,vmin=color_vmin, vmax=color_vmax, interpolation="none", alpha=0.75)

#位置確認
place = np.ones((160, 188))
place[[[height1, height2, height3], [width1, width2, width3]]] = 0
place_mask = np.ma.masked_where(place, place)
ax_im.imshow(place_mask[height_margin1:-height_margin2,:-width_margin], cmap="gray",interpolation="none")

plt.text(width1-4, height1+4-height_margin, "×", color="#55A868", fontsize=48)
plt.text(width2-4, height2+4-height_margin, "×", color="#4C72B0", fontsize=48)
plt.text(width3-4, height3+4-height_margin, "×", color="#C44E52", fontsize=48)
plt.text(width1+5, height1-height_margin, "A", color="#55A868", fontsize=32, weight="bold")
plt.text(width2+5, height2-height_margin, "B", color="#4C72B0", fontsize=32, weight="bold")
plt.text(width3+5, height3-height_margin, "C", color="#C44E52", fontsize=32, weight="bold")
plt.text(width11-4, height11+4-height_margin1, "×", color="#1A1A1A", fontsize=48)
plt.text(w_disp, h_disp-height_margin1-height_margin2, disp_time, color="white", fontsize=32)
plt.text(w_disp+30, h_disp-height_margin1-height_margin2, "s", color="white", fontsize=32)
plt.xticks(visible=False)
plt.yticks(visible=False)
plt.gca().set_axis_off()

divider = make_axes_locatable(ax_im)
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
fig.add_axes(ax_cb)
#plt.colorbar(im, cax=ax_cb).set_label(label="ΔF/F0 (%)",size=18)
plt.colorbar(im, cax=ax_cb)
ax_cb.tick_params(labelsize=22)
plt.setp(plt.getp(ax_cb, 'yticklabels'), color='#1A1A1A')

###fig_right_1
ax=fig.add_subplot(gs[1:4,1:])
ax.axhspan(threshold_down, threshold_up, facecolor="gray", alpha=0.3)
ax.axhline(y=0, color="white", linewidth=3)
ax.axvline(x=0, color="white", linewidth=3)
ax.axvline(x=now, linestyle="--", color="black", linewidth=2)
ax.plot(sound2.time-0.5, sound2.response[:,height1, width1], linewidth=3, label="site A", color="#55A868")
ax.plot(sound2.time-0.5, sound2.response[:,height2, width2], linewidth=3, label="site B", color="#4C72B0")
ax.plot(sound2.time-0.5, sound2.response[:,height3, width3], linewidth=3, label="site C", color="#C44E52")
ax.legend(fontsize=ticks_size, loc=4)
ax.set_xlabel("Time (s)", fontsize=26, color="#1A1A1A")
plt.xticks(fontsize=22, color="#1A1A1A")
#ax.set_ylabel("$\Delta$$F/$$F_0$", fontsize=18)
ax.set_ylabel("ΔF/F0 (%)", fontsize=26, color="#1A1A1A")
plt.yticks(fontsize=22, color="#1A1A1A")
ax.set_ylim(color_vmin, color_vmax)

gs.tight_layout(fig)
#gs.update(hspace=0.3)
#plt.savefig("1216-4kHz_IEEE_frame20.pdf", dpi=600)


# In[14]:

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

dt=0.05

frame=20
threshold_up=0.3
threshold_down=-0.3
color_vmin=-0.8
color_vmax=0.8
colormap=color.bv_color_non_gray

height_margin=40
width_margin=20

height1=89
width1=107

height2=119
width2=88

height3=113
width3=33

now = round(dt*frame-0.5, 2)
disp_time=str(now)
h_disp = 150+5
w_disp = 140-width_margin

fig = plt.figure(figsize=(15,8))
#背景の透明化
fig.patch.set_alpha(0.)
gs = gridspec.GridSpec(5,2)

###fig_left
ax_im = fig.add_subplot(gs[:,0])
#BV_anaのように背景データを線形変換する
#background dataはint16なので演算過程でオーバーフローする->int64に前もって変換
streched = TTX.background.astype(np.int64) / TTX.n_average
#規格化した背景データを描画
normed = streched /streched.max()
#gamma補正を背景に施す
gammaed = normed
ax_im.imshow(gammaed[height_margin:,:-width_margin], vmin=scoreatpercentile(gammaed, 1), vmax=scoreatpercentile(gammaed, 90), cmap=cm.gray, interpolation="none")

#蛍光強度変化データに関して閾値以下の変化を示す蛍光強度変化配列をマスク
up = TTX.response[frame] < threshold_up
down = TTX.response[frame] > threshold_down
condition = up & down
masked = np.ma.masked_where(condition, TTX.response[frame])
#カラーマップをBVanaに合わせるなら、
#import BVcolormap ->cmap=BVcolormap.bv_color
#と指定する
im = ax_im.imshow(masked[height_margin:,:-width_margin],cmap=colormap,vmin=color_vmin, vmax=color_vmax, interpolation="none", alpha=0.75)

plt.text(width1-4, height1+4-height_margin, "×", color="#55A868", fontsize=48)
plt.text(width2-4, height2+4-height_margin, "×", color="#4C72B0", fontsize=48)
plt.text(width3-4, height3+4-height_margin, "×", color="#C44E52", fontsize=48)
plt.text(width1+5, height1-height_margin, "A", color="#55A868", fontsize=32)
plt.text(width2+5, height2-height_margin, "B", color="#4C72B0", fontsize=32)
plt.text(width3+5, height3-height_margin, "C", color="#C44E52", fontsize=32)

plt.text(w_disp, h_disp-height_margin, disp_time, color="white", fontsize=32)
plt.text(w_disp+30, h_disp-height_margin, "s", color="white", fontsize=32)

plt.xticks(visible=False)
plt.yticks(visible=False)
plt.gca().set_axis_off()

divider = make_axes_locatable(ax_im)
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
fig.add_axes(ax_cb)
#plt.colorbar(im, cax=ax_cb).set_label(label="ΔF/F0 (%)",size=18)
plt.colorbar(im, cax=ax_cb)
ax_cb.tick_params(labelsize=22)
plt.setp(plt.getp(ax_cb, 'yticklabels'), color='#1A1A1A')

###fig_right_1
ax=fig.add_subplot(gs[1:4,1:])
ax.axhspan(threshold_down, threshold_up, facecolor="gray", alpha=0.3)
ax.axhline(y=0, color="white", linewidth=3)
ax.axvline(x=0, color="white", linewidth=3)
ax.axvline(x=now, linestyle="--", color="black", linewidth=2)
ax.plot(TTX.time-0.5, TTX.response[:,height1, width1], linewidth=3, label="site A", color="#55A868")
ax.plot(TTX.time-0.5, TTX.response[:,height2, width2], linewidth=3, label="site B", color="#4C72B0")
ax.plot(TTX.time-0.5, TTX.response[:,height3, width3], linewidth=3, label="site C", color="#C44E52")
ax.legend(fontsize=22)
ax.set_xlabel("time (s)", fontsize=26, color="#1A1A1A")
plt.xticks(fontsize=22, color="#1A1A1A")
#ax.set_ylabel("$\Delta$$F/$$F_0$", fontsize=18)
ax.set_ylabel("ΔF/F0 (%)", fontsize=26, color="#1A1A1A")
plt.yticks(fontsize=22, color="#1A1A1A")
ax.set_ylim(color_vmin, color_vmax)

gs.tight_layout(fig)
#gs.update(hspace=0.3)
#plt.savefig("1537MultiA8_IEEE_frame20.pdf", dpi=600)


# #修正後のパラメータ
# ##plt.textに+/-で与えている微調整が変化している

# In[91]:

##
# 位置確認用->位置補正済み
# いままでは波形として表示していたピクセルと×がずれていた
# 今までのheight, width指定で実際に見ていたのはここ
##

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

dt=0.05

ticks_size=22

frame=20
threshold_up=0.2
threshold_down=-0.2
color_vmin=-0.7
color_vmax=0.7
colormap=color.bv_color_non_gray

height_margin=40
width_margin=20

height1=89
width1=107

height2=119
width2=88

height3=113
width3=33

now = round(dt*frame-0.5, 2)
disp_time=str(now)
h_disp = 150+5
w_disp = 140-width_margin

fig = plt.figure(figsize=(15,8))
#背景の透明化
fig.patch.set_alpha(0.)
gs = gridspec.GridSpec(5,2)

###fig_left
ax_im = fig.add_subplot(gs[:,0])
#BV_anaのように背景データを線形変換する
#background dataはint16なので演算過程でオーバーフローする->int64に前もって変換
streched = data.background.astype(np.int64) / data.n_average
#規格化した背景データを描画
normed = streched /streched.max()
#gamma補正を背景に施す
gammaed = normed
ax_im.imshow(gammaed[height_margin:,:-width_margin], vmin=scoreatpercentile(gammaed, 1), vmax=scoreatpercentile(gammaed, 90), cmap=cm.gray, interpolation="none")

#蛍光強度変化データに関して閾値以下の変化を示す蛍光強度変化配列をマスク
up = data.response[frame] < threshold_up
down = data.response[frame] > threshold_down
condition = up & down
masked = np.ma.masked_where(condition, data.response[frame])
#カラーマップをBVanaに合わせるなら、
#import BVcolormap ->cmap=BVcolormap.bv_color
#と指定する
im = ax_im.imshow(masked[height_margin:,:-width_margin],cmap=colormap,vmin=color_vmin, vmax=color_vmax, interpolation="none", alpha=0.75)

#位置確認
place = np.ones((160, 188))
place[[[height1, height2, height3], [width1, width2, width3]]] = 0
place_mask = np.ma.masked_where(place, place)
ax_im.imshow(place_mask[height_margin:,:-width_margin], cmap="gray",interpolation="none")

plt.text(width1-6, height1+6-height_margin, "×", color="#55A868", fontsize=48)
plt.text(width2-6, height2+6-height_margin, "×", color="#4C72B0", fontsize=48)
plt.text(width3-6, height3+6-height_margin, "×", color="#C44E52", fontsize=48)
plt.text(width1+5, height1-height_margin, "A", color="#55A868", fontsize=32, weight="bold")
plt.text(width2+5, height2-height_margin, "B", color="#4C72B0", fontsize=32, weight="bold")
plt.text(width3+5, height3-height_margin, "C", color="#C44E52", fontsize=32, weight="bold")
plt.text(w_disp, h_disp-height_margin, disp_time, color="white", fontsize=32)
plt.text(w_disp+30, h_disp-height_margin, "s", color="white", fontsize=32)
plt.xticks(visible=False)
plt.yticks(visible=False)
plt.gca().set_axis_off()

divider = make_axes_locatable(ax_im)
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
fig.add_axes(ax_cb)
#plt.colorbar(im, cax=ax_cb).set_label(label="ΔF/F0 (%)",size=18)
plt.colorbar(im, cax=ax_cb)
ax_cb.tick_params(labelsize=ticks_size)
plt.setp(plt.getp(ax_cb, 'yticklabels'), color='#1A1A1A')

###fig_right_1
ax=fig.add_subplot(gs[1:4,1:])
ax.axhspan(threshold_down, threshold_up, facecolor="gray", alpha=0.3)
ax.axhline(y=0, color="white", linewidth=3)
ax.axvline(x=0, color="white", linewidth=3)
ax.axvline(x=now, linestyle="--", color="black", linewidth=2)
ax.plot(data.time-0.5, data.response[:,height1, width1], linewidth=3, label="site A", color="#55A868")
ax.plot(data.time-0.5, data.response[:,height2, width2], linewidth=3, label="site B", color="#4C72B0")
ax.plot(data.time-0.5, data.response[:,height3, width3], linewidth=3, label="site C", color="#C44E52")
ax.legend(fontsize=ticks_size, loc=4)
ax.set_xlabel("Time (s)", fontsize=26, color="#1A1A1A")
plt.xticks(fontsize=ticks_size, color="#1A1A1A")
#ax.set_ylabel("$\Delta$$F/$$F_0$", fontsize=18)
ax.set_ylabel("ΔF/F0 (%)", fontsize=26, color="#1A1A1A")
plt.yticks(fontsize=ticks_size, color="#1A1A1A")
ax.set_ylim(color_vmin, color_vmax)

gs.tight_layout(fig)
#gs.update(hspace=0.3)
plt.savefig("1331-500-real_frame20.pdf", dpi=600)


# In[92]:

##
# 位置確認用->位置補正済み
# いままでは波形として表示していたピクセルと×がずれていた
# 今までのheight, width指定で実際に見ていたのはここ
##

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

dt=0.05

ticks_size=22

frame=20
threshold_up=0.2
threshold_down=-0.2
color_vmin=-0.7
color_vmax=0.7
colormap=color.bv_color_non_gray

height_margin=40
width_margin=20

height1=89
width1=107

height2=119
width2=88

height3=113
width3=33

now = round(dt*frame-0.5, 2)
disp_time=str(now)
h_disp = 150+5
w_disp = 140-width_margin

fig = plt.figure(figsize=(15,8))
#背景の透明化
fig.patch.set_alpha(0.)
gs = gridspec.GridSpec(5,2)

###fig_left
ax_im = fig.add_subplot(gs[:,0])
#BV_anaのように背景データを線形変換する
#background dataはint16なので演算過程でオーバーフローする->int64に前もって変換
streched = data.background.astype(np.int64) / data.n_average
#規格化した背景データを描画
normed = streched /streched.max()
#gamma補正を背景に施す
gammaed = normed
ax_im.imshow(gammaed[height_margin:,:-width_margin], vmin=scoreatpercentile(gammaed, 1), vmax=scoreatpercentile(gammaed, 90), cmap=cm.gray, interpolation="none")

#蛍光強度変化データに関して閾値以下の変化を示す蛍光強度変化配列をマスク
up = data.response[frame] < threshold_up
down = data.response[frame] > threshold_down
condition = up & down
masked = np.ma.masked_where(condition, data.response[frame])
#カラーマップをBVanaに合わせるなら、
#import BVcolormap ->cmap=BVcolormap.bv_color
#と指定する
im = ax_im.imshow(masked[height_margin:,:-width_margin],cmap=colormap,vmin=color_vmin, vmax=color_vmax, interpolation="none", alpha=0.75)

#位置確認
place = np.ones((160, 188))
place[[[height1, height2, height3], [width1, width2, width3]]] = 0
place_mask = np.ma.masked_where(place, place)
ax_im.imshow(place_mask[height_margin:,:-width_margin], cmap="gray",interpolation="none")

"""
plt.text(width1-6, height1+6-height_margin, "×", color="#55A868", fontsize=48)
plt.text(width2-6, height2+6-height_margin, "×", color="#4C72B0", fontsize=48)
plt.text(width3-6, height3+6-height_margin, "×", color="#C44E52", fontsize=48)
plt.text(width1+5, height1-height_margin, "A", color="#55A868", fontsize=32, weight="bold")
plt.text(width2+5, height2-height_margin, "B", color="#4C72B0", fontsize=32, weight="bold")
plt.text(width3+5, height3-height_margin, "C", color="#C44E52", fontsize=32, weight="bold")
plt.text(w_disp, h_disp-height_margin, disp_time, color="white", fontsize=32)
plt.text(w_disp+30, h_disp-height_margin, "s", color="white", fontsize=32)
"""
plt.xticks(visible=False)
plt.yticks(visible=False)
plt.gca().set_axis_off()

divider = make_axes_locatable(ax_im)
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
fig.add_axes(ax_cb)
#plt.colorbar(im, cax=ax_cb).set_label(label="ΔF/F0 (%)",size=18)
plt.colorbar(im, cax=ax_cb)
ax_cb.tick_params(labelsize=ticks_size)
plt.setp(plt.getp(ax_cb, 'yticklabels'), color='#1A1A1A')

###fig_right_1
ax=fig.add_subplot(gs[1:4,1:])
ax.axhspan(threshold_down, threshold_up, facecolor="gray", alpha=0.3)
ax.axhline(y=0, color="white", linewidth=3)
ax.axvline(x=0, color="white", linewidth=3)
ax.axvline(x=now, linestyle="--", color="black", linewidth=2)
ax.plot(data.time-0.5, data.response[:,height1, width1], linewidth=3, label="site A", color="#55A868")
ax.plot(data.time-0.5, data.response[:,height2, width2], linewidth=3, label="site B", color="#4C72B0")
ax.plot(data.time-0.5, data.response[:,height3, width3], linewidth=3, label="site C", color="#C44E52")
ax.legend(fontsize=ticks_size, loc=4)
ax.set_xlabel("Time (s)", fontsize=26, color="#1A1A1A")
plt.xticks(fontsize=ticks_size, color="#1A1A1A")
#ax.set_ylabel("$\Delta$$F/$$F_0$", fontsize=18)
ax.set_ylabel("ΔF/F0 (%)", fontsize=26, color="#1A1A1A")
plt.yticks(fontsize=ticks_size, color="#1A1A1A")
ax.set_ylim(color_vmin, color_vmax)

gs.tight_layout(fig)
#gs.update(hspace=0.3)
#plt.savefig("1331-500-dot_frame20.pdf", dpi=600)


# #以降、IEEE用に全て生成しなおし(170106)

# In[138]:

##
# 位置確認用->位置補正済み
# いままでは波形として表示していたピクセルと×がずれていた
# 今までのheight, width指定で実際に見ていたのはここ
##

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

dt=0.05

ticks_size=22

frame=20
threshold_up=0.2
threshold_down=-0.2
color_vmin=-0.7
color_vmax=0.7
colormap=color.bv_color_non_gray

height_margin=40
width_margin=20

height1=89
width1=107

height2=119
width2=88

height3=113
width3=33

now = round(dt*frame-0.5, 2)
disp_time=str(now)
h_disp = 150+5
w_disp = 140-width_margin

fig = plt.figure(figsize=(15,8))
#背景の透明化
fig.patch.set_alpha(0.)
gs = gridspec.GridSpec(5,2)

###fig_left
ax_im = fig.add_subplot(gs[:,0])
#BV_anaのように背景データを線形変換する
#background dataはint16なので演算過程でオーバーフローする->int64に前もって変換
streched = data.background.astype(np.int64) / data.n_average
#規格化した背景データを描画
normed = streched /streched.max()
#gamma補正を背景に施す
gammaed = normed
ax_im.imshow(gammaed[height_margin:,:-width_margin], vmin=scoreatpercentile(gammaed, 1), vmax=scoreatpercentile(gammaed, 90), cmap=cm.gray, interpolation="none")

#蛍光強度変化データに関して閾値以下の変化を示す蛍光強度変化配列をマスク
up = data.response[frame] < threshold_up
down = data.response[frame] > threshold_down
condition = up & down
masked = np.ma.masked_where(condition, data.response[frame])
#カラーマップをBVanaに合わせるなら、
#import BVcolormap ->cmap=BVcolormap.bv_color
#と指定する
im = ax_im.imshow(masked[height_margin:,:-width_margin],cmap=colormap,vmin=color_vmin, vmax=color_vmax, interpolation="none", alpha=0.75)

plt.text(width1-6, height1+6-height_margin, "×", color="#55A868", fontsize=48)
plt.text(width2-6, height2+6-height_margin, "×", color="#4C72B0", fontsize=48)
plt.text(width3-6, height3+6-height_margin, "×", color="#C44E52", fontsize=48)
plt.text(width1+3, height1-height_margin, "A", color="#55A868", fontsize=32, weight="bold")
plt.text(width2+3, height2-height_margin, "B", color="#4C72B0", fontsize=32, weight="bold")
plt.text(width3+3, height3-height_margin, "C", color="#C44E52", fontsize=32, weight="bold")
plt.text(w_disp, h_disp-height_margin, disp_time, color="white", fontsize=32)
plt.text(w_disp+30, h_disp-height_margin, "s", color="white", fontsize=32)
plt.xticks(visible=False)
plt.yticks(visible=False)
plt.gca().set_axis_off()

divider = make_axes_locatable(ax_im)
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
fig.add_axes(ax_cb)
#plt.colorbar(im, cax=ax_cb).set_label(label="ΔF/F0 (%)",size=18)
plt.colorbar(im, cax=ax_cb)
ax_cb.tick_params(labelsize=ticks_size)
plt.setp(plt.getp(ax_cb, 'yticklabels'), color='#1A1A1A')

###fig_right_1
ax=fig.add_subplot(gs[1:4,1:])
ax.axhspan(threshold_down, threshold_up, facecolor="gray", alpha=0.3)
ax.axhline(y=0, color="white", linewidth=3)
ax.axvline(x=0, color="white", linewidth=3)
#ax.axvline(x=now, linestyle="--", color="black", linewidth=2)
ax.plot(data.time-0.5, data.response[:,height1, width1], linewidth=3, label="site A", color="#55A868")
ax.plot(data.time-0.5, data.response[:,height2, width2], linewidth=3, label="site B", color="#4C72B0")
ax.plot(data.time-0.5, data.response[:,height3, width3], linewidth=3, label="site C", color="#C44E52")
ax.legend(fontsize=ticks_size, loc=4)
ax.set_xlabel("Time (s)", fontsize=26, color="#1A1A1A")
plt.xticks(fontsize=ticks_size, color="#1A1A1A")
#ax.set_ylabel("$\Delta$$F/$$F_0$", fontsize=18)
ax.set_ylabel("ΔF/F0 (%)", fontsize=26, color="#1A1A1A")
plt.yticks(fontsize=ticks_size, color="#1A1A1A")
ax.set_ylim(color_vmin, color_vmax)

gs.tight_layout(fig)
#gs.update(hspace=0.3)
plt.savefig("170106BeforTTX-1331-500-frame20.pdf", dpi=600)


# In[136]:

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

dt=0.05

frame=20
threshold_up=0.3
threshold_down=-0.3
color_vmin=-0.7
color_vmax=0.7
colormap=color.bv_color_non_gray

height_margin=40
width_margin=20

height1=99
width1=110

height2=125
width2=93

height3=120
width3=38

now = round(dt*frame-0.5, 2)
disp_time=str(now)
h_disp = 150+5
w_disp = 140-width_margin

fig = plt.figure(figsize=(15,8))
#背景の透明化
fig.patch.set_alpha(0.)
gs = gridspec.GridSpec(5,2)

###fig_left
ax_im = fig.add_subplot(gs[:,0])
#BV_anaのように背景データを線形変換する
#background dataはint16なので演算過程でオーバーフローする->int64に前もって変換
streched = TTX.background.astype(np.int64) / TTX.n_average
#規格化した背景データを描画
normed = streched /streched.max()
#gamma補正を背景に施す
gammaed = normed
ax_im.imshow(gammaed[height_margin:,:-width_margin], vmin=scoreatpercentile(gammaed, 1), vmax=scoreatpercentile(gammaed, 90), cmap=cm.gray, interpolation="none")

#蛍光強度変化データに関して閾値以下の変化を示す蛍光強度変化配列をマスク
up = TTX.response[frame] < threshold_up
down = TTX.response[frame] > threshold_down
condition = up & down
masked = np.ma.masked_where(condition, TTX.response[frame])
#カラーマップをBVanaに合わせるなら、
#import BVcolormap ->cmap=BVcolormap.bv_color
#と指定する
im = ax_im.imshow(masked[height_margin:,:-width_margin],cmap=colormap,vmin=color_vmin, vmax=color_vmax, interpolation="none", alpha=0.75)

#位置確認
place = np.ones((160, 188))
place[[[height1, height2, height3], [width1, width2, width3]]] = 0
place_mask = np.ma.masked_where(place, place)
ax_im.imshow(place_mask[height_margin:,:-width_margin], cmap="gray",interpolation="none")

plt.text(width1-6, height1+6-height_margin, "×", color="#55A868", fontsize=48)
plt.text(width2-6, height2+6-height_margin, "×", color="#4C72B0", fontsize=48)
plt.text(width3-6, height3+6-height_margin, "×", color="#C44E52", fontsize=48)
plt.text(width1+3, height1-height_margin, "A", color="#55A868", fontsize=32, weight="bold")
plt.text(width2+3, height2-height_margin, "B", color="#4C72B0", fontsize=32, weight="bold")
plt.text(width3+3, height3-height_margin, "C", color="#C44E52", fontsize=32, weight="bold")
plt.text(w_disp, h_disp-height_margin, disp_time, color="white", fontsize=32)
plt.text(w_disp+30, h_disp-height_margin, "s", color="white", fontsize=32)

plt.xticks(visible=False)
plt.yticks(visible=False)
plt.gca().set_axis_off()

divider = make_axes_locatable(ax_im)
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
fig.add_axes(ax_cb)
#plt.colorbar(im, cax=ax_cb).set_label(label="ΔF/F0 (%)",size=18)
plt.colorbar(im, cax=ax_cb)
ax_cb.tick_params(labelsize=22)
plt.setp(plt.getp(ax_cb, 'yticklabels'), color='#1A1A1A')

###fig_right_1
ax=fig.add_subplot(gs[1:4,1:])
ax.axhspan(threshold_down, threshold_up, facecolor="gray", alpha=0.3)
ax.axhline(y=0, color="white", linewidth=3)
ax.axvline(x=0, color="white", linewidth=3)
#ax.axvline(x=now, linestyle="--", color="black", linewidth=2)
ax.plot(TTX.time-0.5, TTX.response[:,height1, width1], linewidth=3, label="site A", color="#55A868")
ax.plot(TTX.time-0.5, TTX.response[:,height2, width2], linewidth=3, label="site B", color="#4C72B0")
ax.plot(TTX.time-0.5, TTX.response[:,height3, width3], linewidth=3, label="site C", color="#C44E52")
ax.legend(fontsize=ticks_size, loc=4)
ax.set_xlabel("time (s)", fontsize=26, color="#1A1A1A")
plt.xticks(fontsize=22, color="#1A1A1A")
#ax.set_ylabel("$\Delta$$F/$$F_0$", fontsize=18)
ax.set_ylabel("ΔF/F0 (%)", fontsize=26, color="#1A1A1A")
plt.yticks(fontsize=22, color="#1A1A1A")
#ax.set_ylim(color_vmin, color_vmax)

gs.tight_layout(fig)
#gs.update(hspace=0.3)
plt.savefig("170106AfterTTX-1537MultiA8_IEEE_frame20.pdf", dpi=600)


# In[10]:

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

dt=0.05

frame=20
threshold_up=0.3
threshold_down=-0.3
color_vmin=-0.8
color_vmax=0.8
colormap=color.bv_color_non_gray

height_margin1=20
height_margin2=20
width_margin=20

height1=72
width1=109

height2=88
width2=85

height3=85
width3=37

ticks_size=22

now = round(dt*frame-0.5, 2)
disp_time=str(now)
h_disp = 150+5
w_disp = 140-width_margin

fig = plt.figure(figsize=(15,8))
#背景の透明化
fig.patch.set_alpha(0.)
gs = gridspec.GridSpec(5,2)

###fig_left
ax_im = fig.add_subplot(gs[:,0])
#BV_anaのように背景データを線形変換する
#background dataはint16なので演算過程でオーバーフローする->int64に前もって変換
streched = sound2.background.astype(np.int64) / sound2.n_average
#規格化した背景データを描画
normed = streched /streched.max()
#gamma補正を背景に施す
gammaed = normed
ax_im.imshow(gammaed[height_margin1:-height_margin2,:-width_margin], vmin=scoreatpercentile(gammaed, 1), vmax=scoreatpercentile(gammaed, 90), cmap=cm.gray, interpolation="none")

#蛍光強度変化データに関して閾値以下の変化を示す蛍光強度変化配列をマスク
up = sound2.response[frame] < threshold_up
down = sound2.response[frame] > threshold_down
condition = up & down
masked = np.ma.masked_where(condition, sound2.response[frame])
#カラーマップをBVanaに合わせるなら、
#import BVcolormap ->cmap=BVcolormap.bv_color
#と指定する
im = ax_im.imshow(masked[height_margin1:-height_margin2,:-width_margin],cmap=colormap,vmin=color_vmin, vmax=color_vmax, interpolation="none", alpha=0.75)

#位置確認
place = np.ones((160, 188))
place[[[height1, height2, height3], [width1, width2, width3]]] = 0
place_mask = np.ma.masked_where(place, place)
ax_im.imshow(place_mask[height_margin1:-height_margin2,:-width_margin], cmap="gray",interpolation="none")

plt.text(width1-6, height1+6-height_margin1, "×", color="#55A868", fontsize=48)
plt.text(width2-6, height2+6-height_margin1, "×", color="#4C72B0", fontsize=48)
plt.text(width3-6, height3+6-height_margin1, "×", color="#C44E52", fontsize=48)
plt.text(width1+3, height1-height_margin1, "A", color="#55A868", fontsize=32, weight="bold")
plt.text(width2+3, height2-height_margin1, "B", color="#4C72B0", fontsize=32, weight="bold")
plt.text(width3+3, height3-height_margin1, "C", color="#C44E52", fontsize=32, weight="bold")
#plt.text(width11-4, height11+4-height_margin1, "×", color="#1A1A1A", fontsize=48)
plt.text(w_disp, h_disp-height_margin1-height_margin2, disp_time, color="white", fontsize=32)
plt.text(w_disp+30, h_disp-height_margin1-height_margin2, "s", color="white", fontsize=32)

plt.xticks(visible=False)
plt.yticks(visible=False)
plt.gca().set_axis_off()

divider = make_axes_locatable(ax_im)
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
fig.add_axes(ax_cb)
#plt.colorbar(im, cax=ax_cb).set_label(label="ΔF/F0 (%)",size=18)
plt.colorbar(im, cax=ax_cb)
ax_cb.tick_params(labelsize=22)
plt.setp(plt.getp(ax_cb, 'yticklabels'), color='#1A1A1A')

###fig_right_1
ax=fig.add_subplot(gs[1:4,1:])
ax.axhspan(threshold_down, threshold_up, facecolor="gray", alpha=0.3)
ax.axhline(y=0, color="white", linewidth=3)
ax.axvline(x=0, color="white", linewidth=3)
ax.axvline(x=0, linestyle="--", color="black", linewidth=2)
ax.plot(sound2.time-0.5, sound2.response[:,height1, width1], linewidth=3, label="site A", color="#55A868")
ax.plot(sound2.time-0.5, sound2.response[:,height2, width2], linewidth=3, label="site B", color="#4C72B0")
ax.plot(sound2.time-0.5, sound2.response[:,height3, width3], linewidth=3, label="site C", color="#C44E52")
ax.legend(fontsize=ticks_size, loc=4)
ax.set_xlabel("Time (s)", fontsize=26, color="#1A1A1A")
plt.xticks(fontsize=22, color="#1A1A1A")
#ax.set_ylabel("$\Delta$$F/$$F_0$", fontsize=18)
ax.set_ylabel("ΔF/F0 (%)", fontsize=26, color="#1A1A1A")
plt.yticks(fontsize=22, color="#1A1A1A")
ax.set_ylim(color_vmin, color_vmax)

gs.tight_layout(fig)
#gs.update(hspace=0.3)
#plt.savefig("170106Sound-1216-4kHz_IEEE_frame20.pdf", dpi=600)


# In[142]:

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

dt=0.05

frame=20
threshold_up=0.2
threshold_down=-0.2
color_vmin=-0.7
color_vmax=0.7
colormap=color.bv_color_non_gray

height_margin1=20
height_margin2=20
width_margin=20

height1=72
width1=109

height2=88
width2=85

height3=85
width3=37

now = round(dt*frame-0.5, 2)
disp_time=str(now)
h_disp = 150+5
w_disp = 140-width_margin

fig = plt.figure(figsize=(15,8))
#背景の透明化
fig.patch.set_alpha(0.)
gs = gridspec.GridSpec(5,2)

###fig_left
ax_im = fig.add_subplot(gs[:,0])
#BV_anaのように背景データを線形変換する
#background dataはint16なので演算過程でオーバーフローする->int64に前もって変換
streched = sound2.background.astype(np.int64) / sound2.n_average
#規格化した背景データを描画
normed = streched /streched.max()
#gamma補正を背景に施す
gammaed = normed
ax_im.imshow(gammaed[height_margin1:-height_margin2,:-width_margin], vmin=scoreatpercentile(gammaed, 1), vmax=scoreatpercentile(gammaed, 90), cmap=cm.gray, interpolation="none")

#蛍光強度変化データに関して閾値以下の変化を示す蛍光強度変化配列をマスク
up = sound2.response[frame] < threshold_up
down = sound2.response[frame] > threshold_down
condition = up & down
masked = np.ma.masked_where(condition, sound2.response[frame])
#カラーマップをBVanaに合わせるなら、
#import BVcolormap ->cmap=BVcolormap.bv_color
#と指定する
im = ax_im.imshow(masked[height_margin1:-height_margin2,:-width_margin],cmap=colormap,vmin=color_vmin, vmax=color_vmax, interpolation="none", alpha=0.75)

#位置確認
place = np.ones((160, 188))
place[[[height1, height2, height3], [width1, width2, width3]]] = 0
place_mask = np.ma.masked_where(place, place)
ax_im.imshow(place_mask[height_margin1:-height_margin2,:-width_margin], cmap="gray",interpolation="none")

plt.text(width1-6, height1+6-height_margin1, "×", color="#55A868", fontsize=48)
plt.text(width2-6, height2+6-height_margin1, "×", color="#4C72B0", fontsize=48)
plt.text(width3-6, height3+6-height_margin1, "×", color="#C44E52", fontsize=48)
plt.text(width1+3, height1-height_margin1, "A", color="#55A868", fontsize=32, weight="bold")
plt.text(width2+3, height2-height_margin1, "B", color="#4C72B0", fontsize=32, weight="bold")
plt.text(width3+3, height3-height_margin1, "C", color="#C44E52", fontsize=32, weight="bold")
#plt.text(width11-4, height11+4-height_margin1, "×", color="#1A1A1A", fontsize=48)
plt.text(w_disp, h_disp-height_margin1-height_margin2, disp_time, color="white", fontsize=32)
plt.text(w_disp+30, h_disp-height_margin1-height_margin2, "s", color="white", fontsize=32)

plt.xticks(visible=False)
plt.yticks(visible=False)
plt.gca().set_axis_off()

divider = make_axes_locatable(ax_im)
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
fig.add_axes(ax_cb)
#plt.colorbar(im, cax=ax_cb).set_label(label="ΔF/F0 (%)",size=18)
plt.colorbar(im, cax=ax_cb)
ax_cb.tick_params(labelsize=22)
plt.setp(plt.getp(ax_cb, 'yticklabels'), color='#1A1A1A')

###fig_right_1
ax=fig.add_subplot(gs[1:4,1:])
ax.axhspan(threshold_down, threshold_up, facecolor="gray", alpha=0.3)
ax.axhline(y=0, color="white", linewidth=3)
ax.axvline(x=0, color="white", linewidth=3)
ax.axvline(x=0, linestyle="--", color="black", linewidth=2)
ax.plot(sound2.time-0.5, sound2.response[:,height1, width1], linewidth=3, label="site A", color="#55A868")
ax.plot(sound2.time-0.5, sound2.response[:,height2, width2], linewidth=3, label="site B", color="#4C72B0")
ax.plot(sound2.time-0.5, sound2.response[:,height3, width3], linewidth=3, label="site C", color="#C44E52")
#ax.legend(fontsize=ticks_size, loc=4)
ax.set_xlabel("Time (s)", fontsize=26, color="#1A1A1A")
plt.xticks(fontsize=22, color="#1A1A1A")
#ax.set_ylabel("$\Delta$$F/$$F_0$", fontsize=18)
ax.set_ylabel("ΔF/F0 (%)", fontsize=26, color="#1A1A1A")
plt.yticks(fontsize=22, color="#1A1A1A")
ax.set_ylim(color_vmin, color_vmax)

gs.tight_layout(fig)
#gs.update(hspace=0.3)
plt.savefig("170106Sound-WithoutLegend-1216-4kHz_IEEE_frame20.pdf", dpi=600)


# In[8]:

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

dt=0.05

frame=20
threshold_up=0.3
threshold_down=-0.3
color_vmin=-0.8
color_vmax=0.8
colormap=color.bv_color_non_gray

height_margin1=20
height_margin2=20
width_margin=20

height1=72
width1=109

height2=88
width2=85

height3=85
width3=37

now = round(dt*frame-0.5, 2)
disp_time=str(now)
h_disp = 150+5
w_disp = 140-width_margin

fig = plt.figure(figsize=(15,8))
#背景の透明化
fig.patch.set_alpha(0.)
gs = gridspec.GridSpec(5,2)

###fig_left
ax_im = fig.add_subplot(gs[:,0])
#BV_anaのように背景データを線形変換する
#background dataはint16なので演算過程でオーバーフローする->int64に前もって変換
streched = sound2.background.astype(np.int64) / sound2.n_average
#規格化した背景データを描画
normed = streched /streched.max()
#gamma補正を背景に施す
gammaed = normed
ax_im.imshow(gammaed[height_margin1:-height_margin2,:-width_margin], vmin=scoreatpercentile(gammaed, 1), vmax=scoreatpercentile(gammaed, 90), cmap=cm.gray, interpolation="none")

#蛍光強度変化データに関して閾値以下の変化を示す蛍光強度変化配列をマスク
up = sound2.response[frame] < threshold_up
down = sound2.response[frame] > threshold_down
condition = up & down
masked = np.ma.masked_where(condition, sound2.response[frame])
#カラーマップをBVanaに合わせるなら、
#import BVcolormap ->cmap=BVcolormap.bv_color
#と指定する
im = ax_im.imshow(masked[height_margin1:-height_margin2,:-width_margin],cmap=colormap,vmin=color_vmin, vmax=color_vmax, interpolation="none", alpha=0.75)

#位置確認
"""
place = np.ones((160, 188))
place[[[height1, height2, height3], [width1, width2, width3]]] = 0
place_mask = np.ma.masked_where(place, place)
ax_im.imshow(place_mask[height_margin1:-height_margin2,:-width_margin], cmap="gray",interpolation="none")
"""

plt.text(width1-6, height1+6-height_margin1, "×", color="#55A868", fontsize=48)
plt.text(width2-6, height2+6-height_margin1, "×", color="#C44E52", fontsize=48)
plt.text(width3-6, height3+6-height_margin1, "×", color="#4C72B0", fontsize=48)
plt.text(width1+3, height1-height_margin1, "A", color="#55A868", fontsize=32, weight="bold")
plt.text(width2+3, height2-height_margin1, "B", color="#C44E52", fontsize=32, weight="bold")
plt.text(width3+3, height3-height_margin1, "C", color="#4C72B0", fontsize=32, weight="bold")
#plt.text(width11-4, height11+4-height_margin1, "×", color="#1A1A1A", fontsize=48)
plt.text(w_disp, h_disp-height_margin1-height_margin2, disp_time, color="white", fontsize=32)
plt.text(w_disp+30, h_disp-height_margin1-height_margin2, "s", color="white", fontsize=32)

plt.xticks(visible=False)
plt.yticks(visible=False)
plt.gca().set_axis_off()

divider = make_axes_locatable(ax_im)
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
fig.add_axes(ax_cb)
#plt.colorbar(im, cax=ax_cb).set_label(label="ΔF/F0 (%)",size=18)
plt.colorbar(im, cax=ax_cb)
ax_cb.tick_params(labelsize=22)
plt.setp(plt.getp(ax_cb, 'yticklabels'), color='#1A1A1A')

###fig_right_1
ax=fig.add_subplot(gs[1:4,1:])
ax.axhspan(threshold_down, threshold_up, facecolor="gray", alpha=0.2)
ax.axhline(y=0, color="white", linewidth=3)
ax.axvline(x=0, color="white", linewidth=3)
ax.axvline(x=0, linestyle="--", color="black", linewidth=2)
ax.plot(sound2.time-0.5, sound2.response[:,height1, width1], linewidth=3, label="site A", color="#55A868")
ax.plot(sound2.time-0.5, sound2.response[:,height2, width2], linewidth=3, label="site B", color="#C44E52")
ax.plot(sound2.time-0.5, sound2.response[:,height3, width3], linewidth=3, label="site C", color="#4C72B0")
#ax.legend(fontsize=ticks_size, loc=4)
ax.set_xlabel("Time (s)", fontsize=26, color="#1A1A1A")
plt.xticks(fontsize=22, color="#1A1A1A")
#ax.set_ylabel("$\Delta$$F/$$F_0$", fontsize=18)
ax.set_ylabel("ΔF/F0 (%)", fontsize=26, color="#1A1A1A")
plt.yticks(fontsize=22, color="#1A1A1A")
ax.set_ylim(color_vmin, color_vmax)

gs.tight_layout(fig)
#gs.update(hspace=0.3)
plt.savefig("170106Sound-WithoutLegend-1216-4kHz_IEEE_frame20-th03.pdf", dpi=600)


# In[30]:

##
# 位置確認用->位置補正済み
# いままでは波形として表示していたピクセルと×がずれていた
# 今までのheight, width指定で実際に見ていたのはここ
##

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

dt=0.05

ticks_size=22

frame=20
threshold_up=0.3
threshold_down=-0.3
color_vmin=-0.8
color_vmax=0.8
colormap=color.bv_color_non_gray

height_margin=40
width_margin=20

height1=89
width1=107

height2=119
width2=88

height3=113
width3=33

now = round(dt*frame-0.5, 2)
disp_time=str(now)
h_disp = 150+5
w_disp = 140-width_margin

fig = plt.figure(figsize=(15,8))
#背景の透明化
fig.patch.set_alpha(0.)
gs = gridspec.GridSpec(5,2)

###fig_left
ax_im = fig.add_subplot(gs[:,0])
#BV_anaのように背景データを線形変換する
#background dataはint16なので演算過程でオーバーフローする->int64に前もって変換
streched = data.background.astype(np.int64) / data.n_average
#規格化した背景データを描画
normed = streched /streched.max()
#gamma補正を背景に施す
gammaed = normed
ax_im.imshow(gammaed[height_margin:,:-width_margin], vmin=scoreatpercentile(gammaed, 1), vmax=scoreatpercentile(gammaed, 90), cmap=cm.gray, interpolation="none")

#蛍光強度変化データに関して閾値以下の変化を示す蛍光強度変化配列をマスク
up = data.response[frame] < threshold_up
down = data.response[frame] > threshold_down
condition = up & down
masked = np.ma.masked_where(condition, data.response[frame])
#カラーマップをBVanaに合わせるなら、
#import BVcolormap ->cmap=BVcolormap.bv_color
#と指定する
im = ax_im.imshow(masked[height_margin:,:-width_margin],cmap=colormap,vmin=color_vmin, vmax=color_vmax, interpolation="none", alpha=0.75)

plt.text(width1-6, height1+6-height_margin, "×", color="#55A868", fontsize=48)
plt.text(width2-6, height2+6-height_margin, "×", color="#C44E52", fontsize=48)
plt.text(width3-6, height3+6-height_margin, "×", color="#4C72B0", fontsize=48)
plt.text(width1+3, height1-height_margin, "A", color="#55A868", fontsize=32, weight="bold")
plt.text(width2+3, height2-height_margin, "B", color="#C44E52", fontsize=32, weight="bold")
plt.text(width3+3, height3-height_margin, "C", color="#4C72B0", fontsize=32, weight="bold")
plt.text(w_disp, h_disp-height_margin, disp_time, color="white", fontsize=32)
plt.text(w_disp+30, h_disp-height_margin, "s", color="white", fontsize=32)
plt.xticks(visible=False)
plt.yticks(visible=False)
plt.gca().set_axis_off()

divider = make_axes_locatable(ax_im)
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
fig.add_axes(ax_cb)
#plt.colorbar(im, cax=ax_cb).set_label(label="ΔF/F0 (%)",size=18)
plt.colorbar(im, cax=ax_cb)
ax_cb.tick_params(labelsize=ticks_size)
plt.setp(plt.getp(ax_cb, 'yticklabels'), color='#1A1A1A')

###fig_right_1
ax=fig.add_subplot(gs[1:4,1:])
ax.axhspan(threshold_down, threshold_up, facecolor="gray", alpha=0.2)
ax.axhline(y=0, color="white", linewidth=3)
ax.axvline(x=0, color="white", linewidth=3)
#ax.axvline(x=now, linestyle="--", color="black", linewidth=2)
ax.plot(data.time-0.5, data.response[:,height1, width1], linewidth=3, label="site A", color="#55A868")
ax.plot(data.time-0.5, data.response[:,height2, width2], linewidth=3, label="site B", color="#C44E52")
ax.plot(data.time-0.5, data.response[:,height3, width3], linewidth=3, label="site C", color="#4C72B0")
ax.legend(fontsize=ticks_size, loc=4)
ax.set_xlabel("Time (s)", fontsize=26, color="#1A1A1A")
plt.xticks(fontsize=ticks_size, color="#1A1A1A")
#ax.set_ylabel("$\Delta$$F/$$F_0$", fontsize=18)
ax.set_ylabel("ΔF/F0 (%)", fontsize=26, color="#1A1A1A")
plt.yticks(fontsize=ticks_size, color="#1A1A1A")
ax.set_ylim(color_vmin, color_vmax)

gs.tight_layout(fig)
#gs.update(hspace=0.3)
plt.savefig("170106BeforTTX-1331-500-frame20-th03.pdf", dpi=600)


# In[12]:

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

dt=0.05

ticks_size=22

frame=20
threshold_up=0.3
threshold_down=-0.3
color_vmin=-0.8
color_vmax=0.8
colormap=color.bv_color_non_gray

height_margin=40
width_margin=20

height1=99
width1=110

height2=125
width2=93

height3=120
width3=38

now = round(dt*frame-0.5, 2)
disp_time=str(now)
h_disp = 150+5
w_disp = 140-width_margin

fig = plt.figure(figsize=(15,8))
#背景の透明化
fig.patch.set_alpha(0.)
gs = gridspec.GridSpec(5,2)

###fig_left
ax_im = fig.add_subplot(gs[:,0])
#BV_anaのように背景データを線形変換する
#background dataはint16なので演算過程でオーバーフローする->int64に前もって変換
streched = TTX.background.astype(np.int64) / TTX.n_average
#規格化した背景データを描画
normed = streched /streched.max()
#gamma補正を背景に施す
gammaed = normed
ax_im.imshow(gammaed[height_margin:,:-width_margin], vmin=scoreatpercentile(gammaed, 1), vmax=scoreatpercentile(gammaed, 90), cmap=cm.gray, interpolation="none")

#蛍光強度変化データに関して閾値以下の変化を示す蛍光強度変化配列をマスク
up = TTX.response[frame] < threshold_up
down = TTX.response[frame] > threshold_down
condition = up & down
masked = np.ma.masked_where(condition, TTX.response[frame])
#カラーマップをBVanaに合わせるなら、
#import BVcolormap ->cmap=BVcolormap.bv_color
#と指定する
im = ax_im.imshow(masked[height_margin:,:-width_margin],cmap=colormap,vmin=color_vmin, vmax=color_vmax, interpolation="none", alpha=0.75)

#位置確認
"""
place = np.ones((160, 188))
place[[[height1, height2, height3], [width1, width2, width3]]] = 0
place_mask = np.ma.masked_where(place, place)
ax_im.imshow(place_mask[height_margin:,:-width_margin], cmap="gray",interpolation="none")
"""

plt.text(width1-6, height1+6-height_margin, "×", color="#55A868", fontsize=48)
plt.text(width2-6, height2+6-height_margin, "×", color="#C44E52", fontsize=48)
plt.text(width3-6, height3+6-height_margin, "×", color="#4C72B0", fontsize=48)
plt.text(width1+3, height1-height_margin, "A", color="#55A868", fontsize=32, weight="bold")
plt.text(width2+3, height2-height_margin, "B", color="#C44E52", fontsize=32, weight="bold")
plt.text(width3+3, height3-height_margin, "C", color="#4C72B0", fontsize=32, weight="bold")
plt.text(w_disp, h_disp-height_margin, disp_time, color="white", fontsize=32)
plt.text(w_disp+30, h_disp-height_margin, "s", color="white", fontsize=32)

plt.xticks(visible=False)
plt.yticks(visible=False)
plt.gca().set_axis_off()

divider = make_axes_locatable(ax_im)
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
fig.add_axes(ax_cb)
#plt.colorbar(im, cax=ax_cb).set_label(label="ΔF/F0 (%)",size=18)
plt.colorbar(im, cax=ax_cb, ticks=[-0.7,-0.5,-0.3,-0.1,0,0.1,0.3,0.5,0.7])
ax_cb.tick_params(labelsize=22)
plt.setp(plt.getp(ax_cb, 'yticklabels'), color='#1A1A1A')

###fig_right_1
ax=fig.add_subplot(gs[1:4,1:])
ax.axhspan(threshold_down, threshold_up, facecolor="gray", alpha=0.2)
ax.axhline(y=0, color="white", linewidth=3)
ax.axvline(x=0, color="white", linewidth=3)
#ax.axvline(x=now, linestyle="--", color="black", linewidth=2)
ax.plot(TTX.time-0.5, TTX.response[:,height1, width1], linewidth=3, label="site A", color="#55A868")
ax.plot(TTX.time-0.5, TTX.response[:,height2, width2], linewidth=3, label="site B", color="#C44E52")
ax.plot(TTX.time-0.5, TTX.response[:,height3, width3], linewidth=3, label="site C", color="#4C72B0")
ax.legend(fontsize=ticks_size, loc=4)
ax.set_xlabel("Time (s)", fontsize=26, color="#1A1A1A")
plt.xticks(fontsize=22, color="#1A1A1A")
#ax.set_ylabel("$\Delta$$F/$$F_0$", fontsize=18)
ax.set_ylabel("ΔF/F0 (%)", fontsize=26, color="#1A1A1A")
plt.yticks(fontsize=22, color="#1A1A1A")
#ax.set_ylim(color_vmin, color_vmax)

gs.tight_layout(fig)
#gs.update(hspace=0.3)
plt.savefig("170106AfterTTX-1537MultiA8_IEEE_frame20-th03.pdf", dpi=600)


# # 170407 TTX修正

# In[18]:

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

dt=0.05

ticks_size=22

frame=20
threshold_up=0.3
threshold_down=-0.3
color_vmin=-0.8
color_vmax=0.8
colormap=color.bv_color_non_gray

height_margin=40
width_margin=20

height1=99
width1=114

height2=125
width2=93

height3=120
width3=38

now = round(dt*frame-0.5, 2)
disp_time=str(now)
h_disp = 150+5
w_disp = 140-width_margin

fig = plt.figure(figsize=(15,8))
#背景の透明化
fig.patch.set_alpha(0.)
gs = gridspec.GridSpec(5,2)

###fig_left
ax_im = fig.add_subplot(gs[:,0])
#BV_anaのように背景データを線形変換する
#background dataはint16なので演算過程でオーバーフローする->int64に前もって変換
streched = TTX.background.astype(np.int64) / TTX.n_average
#規格化した背景データを描画
normed = streched /streched.max()
#gamma補正を背景に施す
gammaed = normed
ax_im.imshow(gammaed[height_margin:,:-width_margin], vmin=scoreatpercentile(gammaed, 1), vmax=scoreatpercentile(gammaed, 90), cmap=cm.gray, interpolation="none")

#蛍光強度変化データに関して閾値以下の変化を示す蛍光強度変化配列をマスク
up = TTX.response[frame] < threshold_up
down = TTX.response[frame] > threshold_down
condition = up & down
masked = np.ma.masked_where(condition, TTX.response[frame])
#カラーマップをBVanaに合わせるなら、
#import BVcolormap ->cmap=BVcolormap.bv_color
#と指定する
im = ax_im.imshow(masked[height_margin:,:-width_margin],cmap=colormap,vmin=color_vmin, vmax=color_vmax, interpolation="none", alpha=0.75)

#位置確認
"""
place = np.ones((160, 188))
place[[[height1, height2, height3], [width1, width2, width3]]] = 0
place_mask = np.ma.masked_where(place, place)
ax_im.imshow(place_mask[height_margin:,:-width_margin], cmap="gray",interpolation="none")
"""

plt.text(width1-6, height1+6-height_margin, "×", color="#55A868", fontsize=48)
plt.text(width2-6, height2+6-height_margin, "×", color="#C44E52", fontsize=48)
plt.text(width3-6, height3+6-height_margin, "×", color="#4C72B0", fontsize=48)
plt.text(width1+3, height1-height_margin, "A", color="#55A868", fontsize=32, weight="bold")
plt.text(width2+3, height2-height_margin, "B", color="#C44E52", fontsize=32, weight="bold")
plt.text(width3+3, height3-height_margin, "C", color="#4C72B0", fontsize=32, weight="bold")
plt.text(w_disp, h_disp-height_margin, disp_time, color="white", fontsize=32)
plt.text(w_disp+30, h_disp-height_margin, "s", color="white", fontsize=32)

plt.xticks(visible=False)
plt.yticks(visible=False)
plt.gca().set_axis_off()

divider = make_axes_locatable(ax_im)
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
fig.add_axes(ax_cb)
#plt.colorbar(im, cax=ax_cb).set_label(label="ΔF/F0 (%)",size=18)
plt.colorbar(im, cax=ax_cb, ticks=[-0.7,-0.5,-0.3,-0.1,0,0.1,0.3,0.5,0.7])
ax_cb.tick_params(labelsize=22)
plt.setp(plt.getp(ax_cb, 'yticklabels'), color='#1A1A1A')

###fig_right_1
ax=fig.add_subplot(gs[1:4,1:])
ax.axhspan(threshold_down, threshold_up, facecolor="gray", alpha=0.2)
ax.axhline(y=0, color="white", linewidth=3)
ax.axvline(x=0, color="white", linewidth=3)
#ax.axvline(x=now, linestyle="--", color="black", linewidth=2)
ax.plot(TTX.time-0.5, TTX.response[:,height1, width1], linewidth=3, label="site A", color="#55A868")
ax.plot(TTX.time-0.5, TTX.response[:,height2, width2], linewidth=3, label="site B", color="#C44E52")
ax.plot(TTX.time-0.5, TTX.response[:,height3, width3], linewidth=3, label="site C", color="#4C72B0")
ax.legend(fontsize=ticks_size, loc=4)
ax.set_xlabel("Time (s)", fontsize=26, color="#1A1A1A")
plt.xticks(fontsize=22, color="#1A1A1A")
#ax.set_ylabel("$\Delta$$F/$$F_0$", fontsize=18)
ax.set_ylabel("ΔF/F0 (%)", fontsize=26, color="#1A1A1A")
plt.yticks(fontsize=22, color="#1A1A1A")
#ax.set_ylim(color_vmin, color_vmax)

gs.tight_layout(fig)
#gs.update(hspace=0.3)
plt.savefig("170407AfterTTX-1537MultiA8_IEEE_frame20-th03.pdf", dpi=600)


# In[15]:

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

dt=0.05

ticks_size=22

frame=0
threshold_up=0.3
threshold_down=-0.3
color_vmin=-0.8
color_vmax=0.8
colormap=color.bv_color_non_gray

height_margin=40
width_margin=20

height1=99
width1=114

height2=125
width2=93

height3=120
width3=38

now = round(dt*frame-0.5, 2)
disp_time=str(now)
h_disp = 150+5
w_disp = 140-width_margin

fig = plt.figure(figsize=(15,8))
#背景の透明化
fig.patch.set_alpha(0.)
gs = gridspec.GridSpec(5,2)

###fig_left
ax_im = fig.add_subplot(gs[:,0])
#BV_anaのように背景データを線形変換する
#background dataはint16なので演算過程でオーバーフローする->int64に前もって変換
streched = TTX.background.astype(np.int64) / TTX.n_average
#規格化した背景データを描画
normed = streched /streched.max()
#gamma補正を背景に施す
gammaed = normed
ax_im.imshow(gammaed[height_margin:,:-width_margin], vmin=scoreatpercentile(gammaed, 1), vmax=scoreatpercentile(gammaed, 90), cmap=cm.gray, interpolation="none")

#蛍光強度変化データに関して閾値以下の変化を示す蛍光強度変化配列をマスク
up = TTX.response[frame] < threshold_up
down = TTX.response[frame] > threshold_down
condition = up & down
masked = np.ma.masked_where(condition, TTX.response[frame])
#カラーマップをBVanaに合わせるなら、
#import BVcolormap ->cmap=BVcolormap.bv_color
#と指定する
im = ax_im.imshow(masked[height_margin:,:-width_margin],cmap=colormap,vmin=color_vmin, vmax=color_vmax, interpolation="none", alpha=0.75)

#位置確認
"""
place = np.ones((160, 188))
place[[[height1, height2, height3], [width1, width2, width3]]] = 0
place_mask = np.ma.masked_where(place, place)
ax_im.imshow(place_mask[height_margin:,:-width_margin], cmap="gray",interpolation="none")
"""

plt.text(width1-6, height1+6-height_margin, "×", color="#55A868", fontsize=48)
plt.text(width2-6, height2+6-height_margin, "×", color="#C44E52", fontsize=48)
plt.text(width3-6, height3+6-height_margin, "×", color="#4C72B0", fontsize=48)
plt.text(width1+3, height1-height_margin, "A", color="#55A868", fontsize=32, weight="bold")
plt.text(width2+3, height2-height_margin, "B", color="#C44E52", fontsize=32, weight="bold")
plt.text(width3+3, height3-height_margin, "C", color="#4C72B0", fontsize=32, weight="bold")
plt.text(w_disp, h_disp-height_margin, disp_time, color="white", fontsize=32)
plt.text(w_disp+30, h_disp-height_margin, "s", color="white", fontsize=32)

plt.xticks(visible=False)
plt.yticks(visible=False)
plt.gca().set_axis_off()

divider = make_axes_locatable(ax_im)
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
fig.add_axes(ax_cb)
#plt.colorbar(im, cax=ax_cb).set_label(label="ΔF/F0 (%)",size=18)
plt.colorbar(im, cax=ax_cb, ticks=[-0.7,-0.5,-0.3,-0.1,0,0.1,0.3,0.5,0.7])
ax_cb.tick_params(labelsize=22)
plt.setp(plt.getp(ax_cb, 'yticklabels'), color='#1A1A1A')

###fig_right_1
ax=fig.add_subplot(gs[1:4,1:])
ax.axhspan(threshold_down, threshold_up, facecolor="gray", alpha=0.2)
ax.axhline(y=0, color="white", linewidth=3)
ax.axvline(x=0, color="white", linewidth=3)
#ax.axvline(x=now, linestyle="--", color="black", linewidth=2)
ax.plot(TTX.time-0.5, TTX.response[:,height1, width1], linewidth=3, label="site A", color="#55A868")
ax.plot(TTX.time-0.5, TTX.response[:,height2, width2], linewidth=3, label="site B", color="#C44E52")
ax.plot(TTX.time-0.5, TTX.response[:,height3, width3], linewidth=3, label="site C", color="#4C72B0")
ax.legend(fontsize=ticks_size, loc=4)
ax.set_xlabel("Time (s)", fontsize=26, color="#1A1A1A")
plt.xticks(fontsize=22, color="#1A1A1A")
#ax.set_ylabel("$\Delta$$F/$$F_0$", fontsize=18)
ax.set_ylabel("ΔF/F0 (%)", fontsize=26, color="#1A1A1A")
plt.yticks(fontsize=22, color="#1A1A1A")
#ax.set_ylim(color_vmin, color_vmax)

gs.tight_layout(fig)
#gs.update(hspace=0.3)
plt.savefig("170407AfterTTX-1537MultiA8_IEEE_frame00-th03.pdf", dpi=600)


# # 170621 plt.plotで余計なsmoothingがかかっていないことを確認

# In[12]:

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

dt=0.05

ticks_size=22

frame=0
threshold_up=0.3
threshold_down=-0.3
color_vmin=-0.8
color_vmax=0.8
colormap=color.bv_color_non_gray

height_margin=40
width_margin=20

height1=99
width1=114

height2=125
width2=93

height3=120
width3=38

now = round(dt*frame-0.5, 2)
disp_time=str(now)
h_disp = 150+5
w_disp = 140-width_margin

fig = plt.figure(figsize=(15,8))
#背景の透明化
fig.patch.set_alpha(0.)
gs = gridspec.GridSpec(5,2)

ax=fig.add_subplot(gs[1:4,1:])
ax.scatter(TTX.time-0.5, TTX.response[:,height1, width1])
#ax.plot(TTX.time-0.5, TTX.response[:,height1, width1], linewidth=3, label="site A", color="#55A868")
ax.scatter(TTX.time-0.5, TTX.response[:,height2, width2])
#ax.plot(TTX.time-0.5, TTX.response[:,height2, width2], linewidth=3, label="site B", color="#C44E52")
ax.scatter(TTX.time-0.5, TTX.response[:,height3, width3])
#ax.plot(TTX.time-0.5, TTX.response[:,height3, width3], linewidth=3, label="site C", color="#4C72B0")

ax.legend(fontsize=ticks_size, loc=4)
ax.set_xlabel("Time (s)", fontsize=26, color="#1A1A1A")
plt.xticks(fontsize=22, color="#1A1A1A")
#ax.set_ylabel("$\Delta$$F/$$F_0$", fontsize=18)
ax.set_ylabel("ΔF/F0 (%)", fontsize=26, color="#1A1A1A")
plt.yticks(fontsize=22, color="#1A1A1A")

plt.tight_layout()
plt.savefig("170621-scatter-TTX.png", dpi=300)


# In[13]:

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

dt=0.05

ticks_size=22

frame=0
threshold_up=0.3
threshold_down=-0.3
color_vmin=-0.8
color_vmax=0.8
colormap=color.bv_color_non_gray

height_margin=40
width_margin=20

height1=89
width1=107

height2=119
width2=88

height3=113
width3=33

now = round(dt*frame-0.5, 2)
disp_time=str(now)
h_disp = 150+5
w_disp = 140-width_margin

fig = plt.figure(figsize=(15,8))
#背景の透明化
fig.patch.set_alpha(0.)
gs = gridspec.GridSpec(5,2)

ax=fig.add_subplot(gs[1:4,1:])
ax.scatter(data.time-0.5, data.response[:,height1, width1])
#ax.plot(TTX.time-0.5, TTX.response[:,height1, width1], linewidth=3, label="site A", color="#55A868")
ax.scatter(data.time-0.5, data.response[:,height2, width2])
#ax.plot(TTX.time-0.5, TTX.response[:,height2, width2], linewidth=3, label="site B", color="#C44E52")
ax.scatter(data.time-0.5, data.response[:,height3, width3])
#ax.plot(TTX.time-0.5, TTX.response[:,height3, width3], linewidth=3, label="site C", color="#4C72B0")

ax.legend(fontsize=ticks_size, loc=4)
ax.set_xlabel("Time (s)", fontsize=26, color="#1A1A1A")
plt.xticks(fontsize=22, color="#1A1A1A")
#ax.set_ylabel("$\Delta$$F/$$F_0$", fontsize=18)
ax.set_ylabel("ΔF/F0 (%)", fontsize=26, color="#1A1A1A")
plt.yticks(fontsize=22, color="#1A1A1A")

plt.tight_layout()
plt.savefig("170621-scatter-uMS.png", dpi=300)


# In[18]:

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

dt=0.05

ticks_size=22

frame=0
threshold_up=0.3
threshold_down=-0.3
color_vmin=-0.8
color_vmax=0.8
colormap=color.bv_color_non_gray

height_margin=40
width_margin=20

height1=72
width1=109

height2=88
width2=85

height3=85
width3=37

now = round(dt*frame-0.5, 2)
disp_time=str(now)
h_disp = 150+5
w_disp = 140-width_margin

fig = plt.figure(figsize=(15,8))
#背景の透明化
fig.patch.set_alpha(0.)
gs = gridspec.GridSpec(5,2)

ax=fig.add_subplot(gs[1:4,1:])
ax.scatter(sound2.time-0.5, sound2.response[:,height1, width1])
#ax.plot(TTX.time-0.5, TTX.response[:,height1, width1], linewidth=3, label="site A", color="#55A868")
ax.scatter(sound2.time-0.5, sound2.response[:,height2, width2])
#ax.plot(TTX.time-0.5, TTX.response[:,height2, width2], linewidth=3, label="site B", color="#C44E52")
ax.scatter(sound2.time-0.5, sound2.response[:,height3, width3])
#ax.plot(TTX.time-0.5, TTX.response[:,height3, width3], linewidth=3, label="site C", color="#4C72B0")

ax.legend(fontsize=ticks_size, loc=4)
ax.set_xlabel("Time (s)", fontsize=26, color="#1A1A1A")
plt.xticks(fontsize=22, color="#1A1A1A")
#ax.set_ylabel("$\Delta$$F/$$F_0$", fontsize=18)
ax.set_ylabel("ΔF/F0 (%)", fontsize=26, color="#1A1A1A")
plt.yticks(fontsize=22, color="#1A1A1A")

plt.tight_layout()
plt.savefig("170621-scatter-sound.png", dpi=300)


# In[7]:

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

dt=0.05

ticks_size=22

frame=0
threshold_up=0.3
threshold_down=-0.3
color_vmin=-0.8
color_vmax=0.8
colormap=color.bv_color_non_gray

height_margin=40
width_margin=20

height1=99
width1=114

height2=125
width2=93

height3=120
width3=38

now = round(dt*frame-0.5, 2)
disp_time=str(now)
h_disp = 150+5
w_disp = 140-width_margin

fig = plt.figure(figsize=(15,8))
#背景の透明化
fig.patch.set_alpha(0.)
gs = gridspec.GridSpec(5,2)

###fig_left
ax_im = fig.add_subplot(gs[:,0])
#BV_anaのように背景データを線形変換する
#background dataはint16なので演算過程でオーバーフローする->int64に前もって変換
streched = TTX.background.astype(np.int64) / TTX.n_average
#規格化した背景データを描画
normed = streched /streched.max()
#gamma補正を背景に施す
gammaed = normed
ax_im.imshow(gammaed[height_margin:,:-width_margin], vmin=scoreatpercentile(gammaed, 1), vmax=scoreatpercentile(gammaed, 90), cmap=cm.gray, interpolation="none")

#蛍光強度変化データに関して閾値以下の変化を示す蛍光強度変化配列をマスク
up = TTX.response[frame] < threshold_up
down = TTX.response[frame] > threshold_down
condition = up & down
masked = np.ma.masked_where(condition, TTX.response[frame])
#カラーマップをBVanaに合わせるなら、
#import BVcolormap ->cmap=BVcolormap.bv_color
#と指定する
im = ax_im.imshow(masked[height_margin:,:-width_margin],cmap=colormap,vmin=color_vmin, vmax=color_vmax, interpolation="none", alpha=0.75)

#位置確認
"""
place = np.ones((160, 188))
place[[[height1, height2, height3], [width1, width2, width3]]] = 0
place_mask = np.ma.masked_where(place, place)
ax_im.imshow(place_mask[height_margin:,:-width_margin], cmap="gray",interpolation="none")
"""

plt.text(width1-6, height1+6-height_margin, "×", color="#55A868", fontsize=48)
plt.text(width2-6, height2+6-height_margin, "×", color="#C44E52", fontsize=48)
plt.text(width3-6, height3+6-height_margin, "×", color="#4C72B0", fontsize=48)
plt.text(width1+3, height1-height_margin, "A", color="#55A868", fontsize=32, weight="bold")
plt.text(width2+3, height2-height_margin, "B", color="#C44E52", fontsize=32, weight="bold")
plt.text(width3+3, height3-height_margin, "C", color="#4C72B0", fontsize=32, weight="bold")
plt.text(w_disp, h_disp-height_margin, disp_time, color="white", fontsize=32)
plt.text(w_disp+30, h_disp-height_margin, "s", color="white", fontsize=32)

plt.xticks(visible=False)
plt.yticks(visible=False)
plt.gca().set_axis_off()

divider = make_axes_locatable(ax_im)
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
fig.add_axes(ax_cb)
#plt.colorbar(im, cax=ax_cb).set_label(label="ΔF/F0 (%)",size=18)
plt.colorbar(im, cax=ax_cb, ticks=[-0.7,-0.5,-0.3,-0.1,0,0.1,0.3,0.5,0.7])
ax_cb.tick_params(labelsize=22)
plt.setp(plt.getp(ax_cb, 'yticklabels'), color='#1A1A1A')

###fig_right_1
ax=fig.add_subplot(gs[1:4,1:])
ax.axhspan(threshold_down, threshold_up, facecolor="gray", alpha=0.2)
ax.axhline(y=0, color="white", linewidth=3)
ax.axvline(x=0, color="white", linewidth=3)
#ax.axvline(x=now, linestyle="--", color="black", linewidth=2)
ax.plot(TTX.time-0.5, TTX.response[:,height1, width1], linewidth=3, label="site A", color="#55A868", antialiased=False)
ax.plot(TTX.time-0.5, TTX.response[:,height2, width2], linewidth=3, label="site B", color="#C44E52", antialiased=False)
ax.plot(TTX.time-0.5, TTX.response[:,height3, width3], linewidth=3, label="site C", color="#4C72B0", antialiased=False)
ax.legend(fontsize=ticks_size, loc=4)
ax.set_xlabel("Time (s)", fontsize=26, color="#1A1A1A")
plt.xticks(fontsize=22, color="#1A1A1A")
#ax.set_ylabel("$\Delta$$F/$$F_0$", fontsize=18)
ax.set_ylabel("ΔF/F0 (%)", fontsize=26, color="#1A1A1A")
plt.yticks(fontsize=22, color="#1A1A1A")
#ax.set_ylim(color_vmin, color_vmax)

gs.tight_layout(fig)
#gs.update(hspace=0.3)
#plt.savefig("170407AfterTTX-1537MultiA8_IEEE_frame00-th03.pdf", dpi=600)


# In[ ]:



