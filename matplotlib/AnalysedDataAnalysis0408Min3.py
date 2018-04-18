
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')


# In[3]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
from PIL import Image
import statsmodels.api as sm
import seaborn as sns
sns.set_context("poster")
sns.set_style('darkgrid')
#context_lst = ["paper", "notebook", "talk", "poster"]


# In[3]:

data = pd.read_csv("フラビン生データ(ST=20ms)test160408_min3.csv", encoding="shift-jis")
data = data.ix[:, :37]
data["Max Response (%)"] = data["Max Response (%)"] * (-1)


# In[18]:

grouped_MaxResponse_V = data["Max Response (%)"].groupby([data["depth (mm)"], data["amplitude (V)"]])
grouped_MaxResponse_V.mean().plot(yerr=grouped_MaxResponse_V.std()/np.sqrt(grouped_MaxResponse_V.count()), kind="bar")
plt.title("Minimum Response Intensity")
plt.ylabel("(%)")
plt.tight_layout()
plt.savefig("Min3-MaxResponse_depth_amplitude.png", dpi=300)
#plt.savefig("Min3-MaxResponse_depth_amplitude.eps")


# In[6]:

grouped_latency_V = data["latency (μs)"].groupby([data["depth (mm)"], data["amplitude (V)"]])
grouped_latency_V.mean().plot(yerr=grouped_latency_V.std()/np.sqrt(grouped_latency_V.count()), kind="bar")
plt.title("latency")
plt.ylabel("(μs)")
plt.tight_layout()
#plt.savefig("Min3-latency_depth_amplitude.png", dpi=300)
plt.savefig("Min3-latency_depth_amplitude.eps")


# In[7]:

grouped_duration_V = data["duration (μs)"].groupby([data["depth (mm)"], data["amplitude (V)"]])
grouped_duration_V.mean().plot(yerr=grouped_duration_V.std()/np.sqrt(grouped_duration_V.count()), kind="bar")
plt.title("duration")
plt.ylabel("(μs)")
plt.tight_layout()
#plt.savefig("Min3-duration_depth_amplitude.png", dpi=300)
plt.savefig("Min3-duration_depth_amplitude.eps")


# In[8]:

grouped_RF50_V = data["RF50 (pixel)"].groupby([data["depth (mm)"], data["amplitude (V)"]])
grouped_RF50_V.mean().plot(yerr=grouped_RF50_V.std()/np.sqrt(grouped_RF50_V.count()), kind="bar")
plt.title("RF50")
plt.ylabel("(pixel)")
plt.tight_layout()
plt.savefig("Min3-RF50_depth_amplitude.png", dpi=300)


# In[9]:

grouped_RF75_V = data["RF75 (pixel)"].groupby([data["depth (mm)"], data["amplitude (V)"]])
grouped_RF75_V.mean().plot(yerr=grouped_RF75_V.std()/np.sqrt(grouped_RF75_V.count()), kind="bar")
plt.title("RF75")
plt.ylabel("(pixel)")
plt.tight_layout()
plt.savefig("Min3-RF75_depth_amplitude.png", dpi=300)


# In[10]:

grouped_MaxResponse = data["Max Response (%)"].groupby(data["depth (mm)"])
grouped_MaxResponse.mean().plot(yerr=grouped_MaxResponse.std()/np.sqrt(grouped_MaxResponse.count()), kind="bar")
plt.title("Minimum Response Intensity")
plt.ylabel("(%)")
plt.tight_layout()
plt.savefig("Min3-MaxResponse_depth.png", dpi=300)
#plt.savefig("Min3-MaxResponse_depth.eps")


# In[11]:

grouped_latency = data["latency (μs)"].groupby(data["depth (mm)"])
grouped_latency.mean().plot(yerr=grouped_latency.std()/np.sqrt(grouped_latency.count()), kind="bar")
plt.title("latency")
plt.ylabel("(μs)")
plt.tight_layout()
#plt.savefig("Min3-latency_depth.png", dpi=300)
plt.savefig("Min3-latency_depth.eps")


# In[12]:

grouped_duration = data["duration (μs)"].groupby(data["depth (mm)"])
grouped_duration.mean().plot(yerr=grouped_duration.std()/np.sqrt(grouped_duration.count()), kind="bar")
plt.title("duration")
plt.ylabel("(μs)")
plt.tight_layout()
#plt.savefig("Min3-duration_depth.png", dpi=300)
plt.savefig("Min3-duration_depth.eps")


# In[13]:

grouped_RF50 = data["RF50 (pixel)"].groupby(data["depth (mm)"])
grouped_RF50.mean().plot(yerr=grouped_RF50.std()/np.sqrt(grouped_RF50.count()), kind="bar")
plt.title("RF50")
plt.ylabel("(pixel)")
plt.tight_layout()
plt.savefig("Min3-RF50_depth.png", dpi=300)


# In[14]:

grouped_RF75 = data["RF75 (pixel)"].groupby(data["depth (mm)"])
grouped_RF75.mean().plot(yerr=grouped_RF75.std()/np.sqrt(grouped_RF75.count()), kind="bar")
plt.title("RF75")
plt.ylabel("(pixel)")
plt.tight_layout()
plt.savefig("Min3-RF75_depth.png", dpi=300)


# In[14]:

grouped_MaxResponse_f = data["Max Response (%)"].groupby(data["amplitude (V)"])
grouped_MaxResponse_f.mean().plot(yerr=grouped_MaxResponse_f.std()/np.sqrt(grouped_MaxResponse_f.count()), kind="bar")
plt.title("Minimum Response Intensity")
plt.ylabel("(%)")
plt.tight_layout()
#plt.savefig("Min3-MaxResponse_amplitude.png", dpi=300)
plt.savefig("Min3-MaxResponse_amplitude.eps")


# In[15]:

grouped_latency_f = data["latency (μs)"].groupby(data["amplitude (V)"])
grouped_latency_f.mean().plot(yerr=grouped_latency_f.std()/np.sqrt(grouped_latency_f.count()), kind="bar")
plt.title("latency")
plt.ylabel("(μs)")
plt.tight_layout()
#plt.savefig("Min3-latency_amplitude.png", dpi=300)
plt.savefig("Min3-latency_amplitude.eps")


# In[16]:

grouped_duration_f = data["duration (μs)"].groupby(data["amplitude (V)"])
grouped_duration_f.mean().plot(yerr=grouped_duration_f.std()/np.sqrt(grouped_duration_f.count()), kind="bar")
plt.title("duration")
plt.ylabel("(μs)")
plt.tight_layout()
#plt.savefig("Min3-duration_amplitude.png", dpi=300)
plt.savefig("Min3-duration_amplitude.eps")


# In[18]:

grouped_RF50_f = data["RF50 (pixel)"].groupby(data["amplitude (V)"])
grouped_RF50_f.mean().plot(yerr=grouped_RF50_f.std()/np.sqrt(grouped_RF50_f.count()), kind="bar")
plt.title("RF50")
plt.ylabel("(pixel)")
plt.tight_layout()
plt.savefig("Min3-RF50_amplitude.png", dpi=300)


# In[19]:

grouped_RF75_f = data["RF75 (pixel)"].groupby(data["amplitude (V)"])
grouped_RF75_f.mean().plot(yerr=grouped_RF75_f.std()/np.sqrt(grouped_RF75_f.count()), kind="bar")
plt.title("RF75")
plt.ylabel("(pixel)")
plt.tight_layout()
plt.savefig("Min3-RF75_amplitude.png", dpi=300)


# In[20]:

width = 5
height = 3
img = Image.open("./Min3-MaxResponse_depth_amplitude.png", "r")
wsize = img.size[0]
hsize = img.size[1]
canvas = Image.new('RGB', (width*wsize, height*hsize), (255, 255, 255))
canvas.paste(img, (img.size[0]*0, img.size[1]*0))
img = Image.open("./Min3-latency_depth_amplitude.png", "r")
canvas.paste(img, (img.size[0]*1, img.size[1]*0))
img = Image.open("./Min3-duration_depth_amplitude.png", "r")
canvas.paste(img, (img.size[0]*2, img.size[1]*0))
img = Image.open("./Min3-RF50_depth_amplitude.png", "r")
canvas.paste(img, (img.size[0]*3, img.size[1]*0))
img = Image.open("./Min3-RF75_depth_amplitude.png", "r")
canvas.paste(img, (img.size[0]*4, img.size[1]*0))

img = Image.open("./Min3-MaxResponse_depth.png", "r")
canvas.paste(img, (img.size[0]*0, img.size[1]*1))
img = Image.open("./Min3-latency_depth.png", "r")
canvas.paste(img, (img.size[0]*1, img.size[1]*1))
img = Image.open("./Min3-duration_depth.png", "r")
canvas.paste(img, (img.size[0]*2, img.size[1]*1))
img = Image.open("./Min3-RF50_depth.png", "r")
canvas.paste(img, (img.size[0]*3, img.size[1]*1))
img = Image.open("./Min3-RF75_depth.png", "r")
canvas.paste(img, (img.size[0]*4, img.size[1]*1))

img = Image.open("./Min3-MaxResponse_amplitude.png", "r")
canvas.paste(img, (img.size[0]*0, img.size[1]*2))
img = Image.open("./Min3-latency_amplitude.png", "r")
canvas.paste(img, (img.size[0]*1, img.size[1]*2))
img = Image.open("./Min3-duration_amplitude.png", "r")
canvas.paste(img, (img.size[0]*2, img.size[1]*2))
img = Image.open("./Min3-RF50_amplitude.png", "r")
canvas.paste(img, (img.size[0]*3, img.size[1]*2))
img = Image.open("./Min3-RF75_amplitude.png", "r")
canvas.paste(img, (img.size[0]*4, img.size[1]*2))

canvas.save("./Min3-parameter-merged.png", quality=100)


# In[21]:

x = np.array([0.0,0.5,1.0,2.3])
y = np.array(grouped_MaxResponse.mean())

# サンプルの数
nsample = x.size

# おまじない (後で解説)
X = np.column_stack((np.repeat(1, nsample), x))

model = sm.OLS(y,X)
results = model.fit()
results.summary()


# In[22]:

a, b = results.params

# プロットを表示
plt.bar(x, y, yerr=grouped_MaxResponse.std()/np.sqrt(grouped_MaxResponse.count()), width=0.2)
plt.plot(x, a+b*x)


# In[43]:

grouped_MaxResponse_f = data["Max Response (%)"].groupby(data["amplitude (V)"])

x_f = np.arange(0,5.5,0.5)
y_f = np.array(grouped_MaxResponse_f.mean())

# サンプルの数
nsample = x_f.size

# おまじない (後で解説)
X_f = np.column_stack((np.repeat(1, nsample), x_f))

model_f = sm.OLS(y_f,X_f)
results_f = model_f.fit()
results_f.summary()


# In[44]:

a_f, b_f = results_f.params

# プロットを表示
plt.bar(x_f, y_f, yerr=grouped_MaxResponse_f.std()/np.sqrt(grouped_MaxResponse_f.count()), width=0.1)
plt.plot(x_f, a_f+b_f*x_f)


# In[14]:

(data["latency (μs)"][data["amplitude (V)"]>=2.0][data["depth (mm)"]<=1.0].groupby([data["depth (mm)"], data["amplitude (V)"]]).mean())


# In[10]:

(data["latency (μs)"][data["amplitude (V)"]>=2.0][data["depth (mm)"]<=1.0].groupby([data["depth (mm)"], data["amplitude (V)"]]).std()).mean()


# In[12]:

(data["duration (μs)"][data["amplitude (V)"]>=2.0][data["depth (mm)"]<=1.0].groupby([data["depth (mm)"], data["amplitude (V)"]]).mean()).mean()


# In[13]:

(data["duration (μs)"][data["amplitude (V)"]>=2.0][data["depth (mm)"]<=1.0].groupby([data["depth (mm)"], data["amplitude (V)"]]).std()).mean()


# In[34]:

plt.figure(figsize=(16,6))
plt.subplot(131)

grouped_MaxResponse_f = data["Max Response (%)"].groupby(data["amplitude (V)"])
grouped_MaxResponse_f.mean().plot(yerr=grouped_MaxResponse_f.std()/np.sqrt(grouped_MaxResponse_f.count()), kind="bar")
plt.title("Minimum Response Intensity")
plt.ylabel("(%)")

plt.subplot(132)
grouped_latency_f = data["latency (μs)"].groupby(data["amplitude (V)"])
grouped_latency_f.mean().plot(yerr=grouped_latency_f.std()/np.sqrt(grouped_latency_f.count()), kind="bar")
plt.title("latency")
plt.ylabel("(μs)")

plt.subplot(133)
grouped_duration_f = data["duration (μs)"].groupby(data["amplitude (V)"])
grouped_duration_f.mean().plot(yerr=grouped_duration_f.std()/np.sqrt(grouped_duration_f.count()), kind="bar")
plt.title("duration")
plt.ylabel("(μs)")
plt.tight_layout()
#plt.savefig("Min3_amplitude.png", dpi=300)
plt.savefig("Min3_amplitude.eps")


# In[36]:

plt.figure(figsize=(20,6))

plt.subplot(131)
grouped_MaxResponse_V = data["Max Response (%)"].groupby([data["depth (mm)"], data["amplitude (V)"]])
grouped_MaxResponse_V.mean().plot(yerr=grouped_MaxResponse_V.std()/np.sqrt(grouped_MaxResponse_V.count()), kind="bar")
plt.title("Minimum Response Intensity")
plt.ylabel("(%)")

plt.subplot(132)
grouped_latency_V = data["latency (μs)"].groupby([data["depth (mm)"], data["amplitude (V)"]])
grouped_latency_V.mean().plot(yerr=grouped_latency_V.std()/np.sqrt(grouped_latency_V.count()), kind="bar")
plt.title("latency")
plt.ylabel("(μs)")

plt.subplot(133)
grouped_duration_V = data["duration (μs)"].groupby([data["depth (mm)"], data["amplitude (V)"]])
grouped_duration_V.mean().plot(yerr=grouped_duration_V.std()/np.sqrt(grouped_duration_V.count()), kind="bar")
plt.title("duration")
plt.ylabel("(μs)")
plt.tight_layout()
#plt.savefig("Min3_depth_amplitude.png", dpi=300)
plt.savefig("Min3-duration_amplitude.eps")


# In[38]:

plt.figure(figsize=(16,6))
plt.subplot(131)

grouped_MaxResponse = data["Max Response (%)"].groupby(data["depth (mm)"])
grouped_MaxResponse.mean().plot(yerr=grouped_MaxResponse.std()/np.sqrt(grouped_MaxResponse.count()), kind="bar")
plt.title("Minimum Response Intensity")
plt.ylabel("(%)")

plt.subplot(132)
grouped_latency = data["latency (μs)"].groupby(data["depth (mm)"])
grouped_latency.mean().plot(yerr=grouped_latency.std()/np.sqrt(grouped_latency.count()), kind="bar")
plt.title("latency")
plt.ylabel("(μs)")

plt.subplot(133)
grouped_duration = data["duration (μs)"].groupby(data["depth (mm)"])
grouped_duration.mean().plot(yerr=grouped_duration.std()/np.sqrt(grouped_duration.count()), kind="bar")
plt.title("duration")
plt.ylabel("(μs)")
plt.tight_layout()
#plt.savefig("Min3_depth.png", dpi=300)
plt.savefig("Min3_depth.eps")


# In[12]:

plt.figure(figsize=(16,6))
plt.subplot(131)

grouped_MaxResponse_f = data["Max Response (%)"][data["amplitude (V)"]>0].groupby(data["amplitude (V)"])
grouped_MaxResponse_f.mean().plot(yerr=grouped_MaxResponse_f.std()/np.sqrt(grouped_MaxResponse_f.count()), kind="bar")
plt.title("Inverse Peak Intensity")
plt.ylabel("(%)")

plt.subplot(132)
grouped_latency_f = data["latency (μs)"][data["amplitude (V)"]>0].groupby(data["amplitude (V)"])
grouped_latency_f.mean().plot(yerr=grouped_latency_f.std()/np.sqrt(grouped_latency_f.count()), kind="bar")
plt.title("Latency")
plt.ylabel("(μs)")

plt.subplot(133)
grouped_duration_f = data["duration (μs)"][data["amplitude (V)"]>0].groupby(data["amplitude (V)"])
grouped_duration_f.mean().plot(yerr=grouped_duration_f.std()/np.sqrt(grouped_duration_f.count()), kind="bar")
plt.title("Duration")
plt.ylabel("(μs)")
plt.tight_layout()
#plt.savefig("Min3_amplitude_without0.png", dpi=300)
plt.savefig("Min3_amplitude_without0-label.eps")


# In[6]:

#負のピーク全体のlatency 平均　標準誤差
print(data["latency (μs)"].mean(), data["latency (μs)"].std()/np.sqrt(data["latency (μs)"].count()))


# In[7]:

#負のピーク全体のlatency 平均　標準偏差
print(data["latency (μs)"].mean(), data["latency (μs)"].std())


# In[9]:

#負のピーク1.0V以下のlatency 平均　標準誤差
print(data["latency (μs)"][data["amplitude (V)"]<=1.0].mean(), data["latency (μs)"][data["amplitude (V)"]<=1.0].std()/np.sqrt(data["latency (μs)"][data["amplitude (V)"]<=1.0].count()))


# In[10]:

#負のピーク1.5V以上のlatency 平均　標準誤差
print(data["latency (μs)"][data["amplitude (V)"]>=1.5].mean(), data["latency (μs)"][data["amplitude (V)"]>=1.5].std()/np.sqrt(data["latency (μs)"][data["amplitude (V)"]>=1.5].count()))


# In[4]:

data = pd.read_csv("フラビン生データ(ST=20ms)test160408_min3-rewrite.csv", encoding="shift-jis")
data = data.ix[:, :37]
data["Max Response (%)"] = data["Max Response (%)"] * (-1)
data["Peak Latency (ms)"] = data["max時time"]


# In[19]:

plt.figure(figsize=(16,6))
plt.subplot(131)

grouped_MaxResponse_f = data["Max Response (%)"][data["amplitude (V)"]>0].groupby(data["amplitude (V)"])
grouped_MaxResponse_f.mean().plot(yerr=grouped_MaxResponse_f.std()/np.sqrt(grouped_MaxResponse_f.count()), kind="bar")
plt.title("Inverse Peak Intensity")
plt.ylabel("(%)")

plt.subplot(132)
grouped_latency_f = data["Peak Latency (ms)"][data["amplitude (V)"]>0].groupby(data["amplitude (V)"])
grouped_latency_f.mean().plot(yerr=grouped_latency_f.std()/np.sqrt(grouped_latency_f.count()), kind="bar")
plt.title("Peak Latency")
plt.ylabel("(ms)")

plt.subplot(133)
grouped_duration_f = data["duration (μs)"][data["amplitude (V)"]>0].groupby(data["amplitude (V)"])
grouped_duration_f.mean().plot(yerr=grouped_duration_f.std()/np.sqrt(grouped_duration_f.count()), kind="bar")
plt.title("Duration")
plt.ylabel("(ms)")
plt.tight_layout()
#plt.savefig("Min3_amplitude_without0-basedPeak.png", dpi=300)
plt.savefig("Min3_amplitude_without0-basedPeak.eps")


# In[20]:

#負のピーク全体のlatency 平均　標準誤差
print(data["Peak Latency (ms)"].mean(), data["Peak Latency (ms)"].std()/np.sqrt(data["latency (μs)"].count()))


# In[21]:

#負のピーク全体のlatency 平均　標準偏差
print(data["Peak Latency (ms)"].mean(), data["Peak Latency (ms)"].std())


# In[22]:

#負のピーク1.0V以下のlatency 平均　標準誤差
print(data["Peak Latency (ms)"][data["amplitude (V)"]<=1.0].mean(), data["Peak Latency (ms)"][data["amplitude (V)"]<=1.0].std()/np.sqrt(data["Peak Latency (ms)"][data["amplitude (V)"]<=1.0].count()))


# In[23]:

#負のピーク1.5V以上のlatency 平均　標準誤差
print(data["Peak Latency (ms)"][data["amplitude (V)"]>=1.5].mean(), data["Peak Latency (ms)"][data["amplitude (V)"]>=1.5].std()/np.sqrt(data["Peak Latency (ms)"][data["amplitude (V)"]>=1.5].count()))


# In[18]:

plt.figure(figsize=(16,6))
plt.subplot(131)

grouped_MaxResponse_f = data["Max Response (%)"][data["amplitude (V)"]>0].groupby(data["amplitude (V)"])
grouped_MaxResponse_f.mean().plot(marker="o",linestyle="--", yerr=grouped_MaxResponse_f.std()/np.sqrt(grouped_MaxResponse_f.count()))
plt.xlim([0.3,5.2])
plt.xticks([0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0])
plt.title("Inverse Peak Intensity")
plt.ylabel("(%)")

plt.subplot(132)
grouped_latency_f = data["Peak Latency (ms)"][data["amplitude (V)"]>0].groupby(data["amplitude (V)"])
grouped_latency_f.mean().plot(marker="o",linestyle="--", yerr=grouped_latency_f.std()/np.sqrt(grouped_latency_f.count()))
plt.xlim([0.3,5.2])
plt.xticks([0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0])
plt.title("Peak Latency")
plt.ylabel("(ms)")

plt.subplot(133)
grouped_duration_f = data["duration (μs)"][data["amplitude (V)"]>0].groupby(data["amplitude (V)"])
grouped_duration_f.mean().plot(marker="o",linestyle="--", yerr=grouped_duration_f.std()/np.sqrt(grouped_duration_f.count()))
plt.xlim([0.3,5.2])
plt.xticks([0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0])
plt.title("Duration")
plt.ylabel("(ms)")
plt.tight_layout()
plt.savefig("Min3_amplitude_without0-basedPeak-nonbar.png", dpi=300)
plt.savefig("Min3_amplitude_without0-basedPeak-nonbar.eps")


# In[ ]:



