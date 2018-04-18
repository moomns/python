
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')


# In[45]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate


# In[47]:

#コイル
N = 100
r = 0.75*10**(-3)
S = np.pi * r ** 2
fs = 200*10**3
dt = 1./fs
pre = 50


# In[103]:

for i in np.arange(1,33,1):
    tmp = pd.read_csv("data-in_151208_3.0_0.001s_200kHz_"+str(i) +".csv" )
    #induced electromotive force
    tmp.columns=["time[s]", "AI0",str(i),"AI2[V]"]
    
    #offset correction
    dc_noise = tmp[str(i)][60:].mean()
    correction = ((tmp[str(i)]-dc_noise)*dt/(N*S))
    #numerical integration
    tmp2 = integrate.cumtrapz(correction, tmp["time[s]"]*10**6)
    tmp2 = pd.DataFrame(tmp2)
    tmp2.columns = ([str(i)])                  
    
    if i == 1:
        induced = pd.concat([tmp["time[s]"]*10**6, tmp[str(i)]], axis=1)
        integrated = pd.concat([tmp["time[s]"]*10**6, tmp2], axis=1)
        
    else:
        induced = pd.concat([induced, tmp[str(i)]], axis=1)
        integrated = pd.concat([integrated, tmp2], axis=1)


# In[164]:

#不偏標準偏差
induced.std(axis=1, ddof=True)
#標本標準偏差
#induced.std(axis=1, ddof=False)


# In[55]:

#次元のタプル
induced.shape
#列数
induced.shape[1]


# In[104]:

#標準誤差
error_induced = induced.ix[:, 1:].std(axis=1,ddof=True)/np.sqrt(induced.shape[1])
#標準誤差
error_integrated = integrated.ix[:, 1:].std(axis=1,ddof=True)/np.sqrt(integrated.shape[1])


# In[63]:

plt.errorbar(induced["time[s]"], induced.ix[:, 1:].mean(axis=1), yerr=error, fmt='o', capsize=10, ms=10)


# In[64]:

plt.plot(induced["time[s]"], induced.ix[:, 1:].mean(axis=1))


# In[106]:

plt.plot(integrated["time[s]"], integrated.ix[:, 1:].mean(axis=1)*10**3)


# In[147]:

plt.rcParams['font.family'] = 'Arial'
plt.figure(figsize=(16,14))
plt.subplot(121)
plt.plot((induced["time[s]"])[:100], (induced.ix[:, 1:].mean(axis=1)*10**3)[:100], linewidth=2)
plt.grid(linewidth=2, alpha=0.7)
plt.title("Induced electromotive force", fontsize=24)
plt.xlabel("time [μs]", fontsize=24)
plt.xticks(fontsize=24)
plt.ylabel("Voltage [mV]", fontsize=24)
plt.yticks(fontsize=24)

plt.subplot(122)
plt.plot((integrated["time[s]"])[:100], (integrated.ix[:, 1:].mean(axis=1)*10**3)[:100], linewidth=2)
plt.grid(linewidth=2, alpha=0.7)
plt.title("Numeric Integration", fontsize=24)
plt.xlabel("time [μs]", fontsize=24)
plt.xticks(fontsize=24)
plt.ylabel("magnetic flux density [mT]", fontsize=24)
plt.yticks(fontsize=24)
plt.tight_layout()
#plt.savefig("induced-Integrated.png", dpi=180)
plt.savefig("induced-Integrated.eps")
#plt.close()


# In[149]:

plt.rcParams['font.family'] = 'Arial'
fig = plt.figure(figsize=(16,14))
ax1 = fig.add_subplot(121)
ax1.errorbar((induced["time[s]"])[:100], (induced.ix[:, 1:].mean(axis=1)*10**3)[:100], yerr=error_induced[:100]*10**3, fmt='o', capsize=10, ms=8, elinewidth=2, mew=2, mec="g", color="g", alpha=0.5)
#ms:markersize
#elinewidth:width of bar
#mew:maekeredgewidth
#mec:markeredgecolor
ax1.plot((induced["time[s]"])[:100], (induced.ix[:, 1:].mean(axis=1)*10**3)[:100], linewidth=2, color="b")
ax1.grid(linewidth=2, alpha=0.7)
ax1.set_title("Induced electromotive force", fontsize=24)
ax1.set_xlabel("time [μs]", fontsize=24)
ax1.set_xticklabels([0,100,200,300,400,500], fontsize=24)
ax1.set_ylabel("Voltage [mV]", fontsize=24)
ax1.set_yticklabels([-40,-30,-20,-10,0,10,20,30,40],fontsize=24)

ax2=fig.add_subplot(122)
ax2.errorbar((integrated["time[s]"])[:100], (integrated.ix[:, 1:].mean(axis=1)*10**3)[:100], yerr=error_integrated[:100]*10**3, fmt='o', capsize=10, ms=8, elinewidth=2, mew=2, mec="g", color="g", alpha=0.5)
ax2.plot((integrated["time[s]"])[:100], (integrated.ix[:, 1:].mean(axis=1)*10**3)[:100], linewidth=2, color="b")
ax2.grid(linewidth=2, alpha=0.7)
ax2.set_title("Numeric Integration", fontsize=24)
ax2.set_xlabel("time [μs]", fontsize=24)
ax2.set_xticklabels([0,100,200,300,400,500], fontsize=24)
ax2.set_ylabel("magnetic flux density [mT]", fontsize=24)
ax2.set_yticklabels([-6,-5,-4,-3,-2,-1,0,1,2], fontsize=24)
plt.tight_layout()
#plt.savefig("induced-Integrated-Errorbar.png", dpi=180)
plt.savefig("induced-Integrated-Errorbar.eps")
#plt.close()


# In[163]:

#extract index of absmax
integrated.ix[:, 1:].mean(axis=1).abs().argmax()
#display absmax
(integrated.ix[:, 1:].mean(axis=1)*10**3)[36]
#dispray SE of index of absmax
error_integrated[36]*10**3


# In[ ]:



