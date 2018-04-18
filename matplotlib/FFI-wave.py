
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')


# In[2]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[10]:

CtoR = pd.read_csv("1521-3500um-wave-CtoR.csv", skiprows=10)
CtoR = CtoR.ix[:, :-1]


# In[13]:

CtoR


# In[16]:

for i in np.arange(1,9,1):
    plt.plot(CtoR["time"], CtoR["value"+str(i)], color="black", alpha=(10-i)*0.1)
    plt.legend()
    plt.grid()


# In[19]:

for i in np.arange(1,9,1):
    if i==2:
        pass
    else:
        plt.plot(CtoR["time"], CtoR["value"+str(i)], color="black", alpha=(10-i)*0.1)
plt.legend()
plt.grid()


# In[17]:

DCtoRV = pd.read_csv("1521-3500um-wave-DCtoRV.csv", skiprows=10)
DCtoRV = DCtoRV.ix[:, :-1]

for i in np.arange(1,9,1):
    plt.plot(DCtoRV["time"], DCtoRV["value"+str(i)], color="black", alpha=(10-i)*0.1)
    plt.legend()
    plt.grid()


# In[28]:

fig = plt.figure(figsize=(6,14))
plt.subplots_adjust(hspace=0.001)

ax1 = plt.subplot(711)
ax1.plot(CtoR["time"], CtoR["value2"], label="No.1")
ax1.set_ylim(-0.1, 0.8)
ax1.grid()
ax1.legend()

ax2 = plt.subplot(712, sharex=ax1)
ax2.plot(CtoR["time"], CtoR["value1"], label="No.2")
ax2.set_ylim(-0.1, 0.3)
ax2.grid()
ax2.legend()

ax3 = plt.subplot(713, sharex=ax1)
ax3.plot(CtoR["time"], CtoR["value3"], label="No.3")
ax3.set_ylim(-0.1, 0.3)
ax3.grid()
ax3.legend()

ax4 = plt.subplot(714, sharex=ax1)
ax4.plot(CtoR["time"], CtoR["value4"], label="No.4")
ax4.set_ylim(-0.1, 0.3)
ax4.grid()
ax4.legend()

ax5 = plt.subplot(715, sharex=ax1)
ax5.plot(CtoR["time"], CtoR["value5"], label="No.5")
ax5.set_ylim(-0.1, 0.3)
ax5.grid()
ax5.legend()

ax6 = plt.subplot(716, sharex=ax1)
ax6.plot(CtoR["time"], CtoR["value6"], label="No.6")
ax6.set_ylim(-0.1, 0.3)
ax6.grid()
ax6.legend()

ax7 = plt.subplot(717, sharex=ax1)
ax7.plot(CtoR["time"], CtoR["value7"], label="No.7")
ax7.set_ylim(-0.1, 0.3)
ax7.grid()
ax7.legend()

xticklabels=ax1.get_xticklabels()+ax2.get_xticklabels()+ax3.get_xticklabels()+ax4.get_xticklabels()+ax5.get_xticklabels()+ax6.get_xticklabels()
plt.setp(xticklabels, visible=False)




# In[46]:

fig = plt.figure(figsize=(13,14))
plt.subplots_adjust(hspace=0.1, wspace=0.1)

ax1 = plt.subplot(7, 2, 1)
ax1.plot(CtoR["time"], CtoR["value2"], label="No.1")
ax1.set_ylim(-0.1, 0.8)
ax1.grid()
ax1.legend()

ax2 = plt.subplot(7, 2, 3, sharex=ax1)
ax2.plot(CtoR["time"], CtoR["value1"], label="No.2")
ax2.set_ylim(-0.1, 0.3)
ax2.grid()
ax2.legend()

ax3 = plt.subplot(7, 2, 5, sharex=ax1)
ax3.plot(CtoR["time"], CtoR["value3"], label="No.3")
ax3.set_ylim(-0.1, 0.3)
ax3.grid()
ax3.legend()

ax4 = plt.subplot(7, 2, 7, sharex=ax1)
ax4.plot(CtoR["time"], CtoR["value4"], label="No.4")
ax4.set_ylim(-0.1, 0.3)
ax4.grid()
ax4.legend()

ax5 = plt.subplot(7, 2, 9, sharex=ax1)
ax5.plot(CtoR["time"], CtoR["value5"], label="No.5")
ax5.set_ylim(-0.1, 0.3)
ax5.grid()
ax5.legend()

ax6 = plt.subplot(7, 2, 11, sharex=ax1)
ax6.plot(CtoR["time"], CtoR["value6"], label="No.6")
ax6.set_ylim(-0.1, 0.3)
ax6.grid()
ax6.legend()

ax7 = plt.subplot(7, 2, 13, sharex=ax1)
ax7.plot(CtoR["time"], CtoR["value7"], label="No.7")
ax7.set_ylim(-0.1, 0.3)
ax7.set_xlabel("time [ms]")
ax7.grid()
ax7.legend()

xticklabels=ax1.get_xticklabels()+ax2.get_xticklabels()+ax3.get_xticklabels()+ax4.get_xticklabels()+ax5.get_xticklabels()+ax6.get_xticklabels()
plt.setp(xticklabels, visible=False)

ax8 = plt.subplot(7, 2, 2)
ax8.plot(DCtoRV["time"], DCtoRV["value3"], label="A")
ax8.set_ylim(-0.2, 2.5)
ax8.grid()
ax8.legend()

ax9 = plt.subplot(7, 2, 4, sharex=ax8)
ax9.plot(DCtoRV["time"], DCtoRV["value4"], label="B")
ax9.set_ylim(-1.2, 2.5)
ax9.grid()
ax9.legend()

ax10 = plt.subplot(7, 2, 6, sharex=ax8)
ax10.plot(DCtoRV["time"], DCtoRV["value5"], label="C")
ax10.set_ylim(-0.2, 2.5)
ax10.grid()
ax10.legend()

ax11 = plt.subplot(7, 2, 8, sharex=ax8)
ax11.plot(DCtoRV["time"], DCtoRV["value6"], label="D")
ax11.set_ylim(-0.2, 2.5)
ax11.grid()
ax11.legend()

ax12 = plt.subplot(7, 2, 10, sharex=ax8)
ax12.plot(DCtoRV["time"], DCtoRV["value7"], label="E")
ax12.set_ylim(-0.2, 2.5)
ax12.grid()
ax12.legend()

ax13 = plt.subplot(7, 2, 12, sharex=ax8)
ax13.plot(DCtoRV["time"], DCtoRV["value8"], label="F")
ax13.set_ylim(-0.2, 2.5)
ax13.set_xlabel("time [ms]")
ax13.grid()
ax13.legend()

xticklabels=ax8.get_xticklabels()+ax9.get_xticklabels()+ax10.get_xticklabels()+ax11.get_xticklabels()+ax12.get_xticklabels()
plt.setp(xticklabels, visible=False)

#plt.tight_layout()
plt.savefig("FFI-wave.png", dpi=180)


# In[ ]:



