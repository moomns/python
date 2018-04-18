from scipy import integrate
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
plt.rcParams['font.family'] = 'Century'

gs=gridspec.GridSpec(1, 3, height_ratios=[1, 1, 1]) 
plt.figure(figsize=(16,10))

data = pd.read_csv("tmp.csv", index_col="index")
ax1=plt.subplot(gs[0])
ax1.imshow(data.ix[:,10:51], interpolation=None)
ax1.set_xticks([0,20,40], [-500,0,500])
ax1.set_xlabel("[um]", fontsize=16)
ax1.set_yticks([0,20,40,60,80], [-1000,-500,0,500,1000])
ax1.set_ylabel("[um]", fontsize=16)
ax1.set_title("Estimation", fontsize=16)
cbar=plt.colorbar(ax1)
cbar.set_label("magnetic flux density[mT]", fontsize=16)
for t in cbar.ax.get_yticklabels():

    t.set_fontsize(16)


N = 100
r = 0.75*10**(-3)
S = np.pi * r ** 2
fs = 200*10**3
dt = 1./fs
pre = 50

data = pd.read_csv("concat.csv")

time = data["time[s]"]*10**6
dc_noise = data["Response[V]"][60:].mean()
correction = ((data["Response[V]"]-dc_noise)*dt/(N*S)) *10**3
integrated = integrate.cumtrapz(correction, time)

ax2=plt.subplot(gs[1])
ax2.plot(time[:100], data["Response[V]"][:100]*10**3)
ax2.set_grid()
ax2.set_title("Induced electromotive force", fontsize=16)
ax2.set_xlabel("time[us]", fontsize=16)
ax2.set_ylabel("Voltage[mV]", fontsize=16)

ax3=plt.subplot(gs[2])
ax3.plot(time[:100], integrated[:100])
ax3.set_grid()
ax3.set_title("Numeric Integration", fontsize=16)
ax3.set_xlabel("time[us]", fontsize=16)
ax3.set_ylabel("magnetic flux density[mT]", fontsize=16)
plt.tight_layout()
plt.savefig("test-size.png", dpi=180)
#plt.savefig("Induced-Integrated.eps")
plt.close()