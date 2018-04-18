
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')


# In[2]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:

data = pd.read_csv("concat.csv")


# In[4]:

plt.figure(figsize=(8,6))

plt.subplot(121)
plt.plot(data["time[s]"][:80]*10**6, data["Response[V]"][:80]*10**3, label="Response", color=sns.xkcd_rgb["denim blue"])
plt.xlabel("time[μs]",fontsize=18)
plt.ylabel("Voltage[mV]",  fontsize=18)
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=18)
#plt.title("Stim(0-3V->Amp) Average32 Above the sensor")
plt.grid(linewidth=1)
plt.legend(loc="upper center", bbox_to_anchor=(0.5,1.12), ncol=2, fontsize=18)

plt.subplot(122)
#plt.plot(data["time[s]"][:80]*10**6, data["Stimulation[V]"][:80], label="Stim", color=sns.xkcd_rgb["medium green"])
plt.plot(data["time[s]"][:80]*10**6, data["Trigger[V]"][:80], label="Trigger", color=sns.xkcd_rgb["pale red"])
plt.xlabel("time[μs]",  fontsize=18)
plt.ylabel("Voltage[V]",  fontsize=18)
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=18)
#plt.title("Stim(0-3V->Amp) Average32 Above the sensor")
plt.grid(linewidth=1)
plt.legend(loc="upper center", bbox_to_anchor=(0.5,1.12), ncol=2, fontsize=18)

plt.tight_layout()

plt.savefig("Res_and_StimTri-Seaborn-0204-NonStim.png", dpi=180)


# In[5]:

#plt.figure(figsize=(12,6))
sns.set_style("darkgrid")
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(data["time[s]"]*10**6, data["Response[V]"]*10**3, label="Response", color="b")
ax1.set_ylabel("Voltage[mV]", fontsize=18)
ax1.set_yticklabels([-20,-15,-10,-5,0,5,10,15,20], fontsize=18)
ax1.set_xlabel("time[us]", fontsize=18)
ax1.set_xticklabels([0,200,400,600,800,1000], fontsize=18)
ax1.grid()
ax2 = ax1.twinx()
ax2.plot(data["time[s]"]*10**6, data["Stimulation[V]"], label="Stimulation", color="g", alpha=0.5)
ax2.plot(data["time[s]"]*10**6, data["Trigger[V]"], label="Trigger", color="r", alpha=0.5)
ax2.set_ylabel("Voltage[V]", fontsize=18)
ax2.set_ylim(-20, 20)
for t in ax2.get_yticklabels():
    t.set_fontsize(18)
ax2.set_title("Stim(0-3V->Amp) Average32 Above the sensor", fontsize=18)
ax2.grid()

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc=4, fontsize=18)

plt.savefig("ResStimTri-Seaborn.png", dpi=180)


# In[6]:

#plt.figure(figsize=(12,6))
sns.set_style("darkgrid")
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylabel("Voltage [mV]", fontsize=18)
ax1.set_ylim(-20, 20)
ax1.set_yticklabels([-20,-15,-10,-5,0,5,10,15,20], fontsize=18)
ax1.set_xlabel("time [μs]", fontsize=18)
ax1.set_xticklabels([0,200,400,600,800,1000], fontsize=18)
ax2 = ax1.twinx()
ax2.plot(data["time[s]"]*10**6, data["Response[V]"]*10**3, label="Response")
ax2.plot(data["time[s]"]*10**6, data["Stimulation[V]"], label="Stimulation")
ax2.plot(data["time[s]"]*10**6, data["Trigger[V]"], label="Trigger")
ax2.set_ylabel("Voltage [V]", fontsize=18)
ax2.set_ylim(-20, 20)
for t in ax2.get_yticklabels():
    t.set_fontsize(18)
ax2.set_title("Stim(0-3 V -> Amp) Average32 apart from the sensor", fontsize=18)
ax2.grid()
ax2.legend(fontsize=18)

"""
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc=4, fontsize=18)
"""
plt.grid()

plt.savefig("ResStimTri-Seaborn2-apart.png", dpi=180)


# In[10]:

data


# In[5]:

#plt.figure(figsize=(12,6))
sns.set_style("darkgrid")
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(data["time[s]"]*10**6, data["Response[V]"]*10**3, label="Response [mV]")
ax1.plot(data["time[s]"]*10**6, data["Stimulation[V]"], label="Stimulation [V]")
ax1.plot(data["time[s]"]*10**6, data["Trigger[V]"], label="Trigger [V]")
ax1.set_ylabel("Voltage", fontsize=18)
ax1.set_ylim(-20, 20)
ax1.set_yticklabels([-20,-15,-10,-5,0,5,10,15,20], fontsize=18)
ax1.set_xlabel("time [μs]", fontsize=18)
ax1.set_xticklabels([0,200,400,600,800,1000], fontsize=18)
ax1.grid()
ax1.set_title("Stim(0-3V->Amp) Average32 Above the sensor", fontsize=18)
ax1.legend(fontsize=18)

"""
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc=4, fontsize=18)
"""
plt.grid()

plt.savefig("ResStimTri-Seaborn3-micro.png", dpi=180)


# In[39]:

plt.rcParams['font.family'] = 'Arial'
fig = plt.figure(figsize=(8,6))
plt.subplots_adjust(hspace=0.1)

ax1=plt.subplot(211)
ax1.plot(data["time[s]"][:80]*10**6, data["Response[V]"][:80]*10**3, label="Response", color=sns.xkcd_rgb["denim blue"])
ax1.set_ylabel("Voltage[mV]",  fontsize=18)
ax1.set_yticks([-8,-4,0,4,8])
ax1.set_yticklabels([-8,-4,0,4,8], fontsize=16)
ax1.grid(linewidth=1)
ax1.legend(fontsize=18)
plt.setp(ax1.get_xticklabels(), visible=False)

ax2=plt.subplot(212, sharex=ax1)
ax2.plot(data["time[s]"][:80]*10**6, data["Trigger[V]"][:80], label="Trigger", color=sns.xkcd_rgb["pale red"])
ax2.set_xlabel("time[μs]",  fontsize=18)
ax2.set_ylabel("Voltage[V]",  fontsize=18)
ax2.set_xticklabels([0,50,100,150,200,250,300,350,400], fontsize=16)
ax2.set_yticks([0,0.4,0.8,1.2])
ax2.set_yticklabels([0,0.4,0.8,1.2],fontsize=16)
ax2.grid(linewidth=1)
ax2.legend(fontsize=18)

plt.tight_layout()

plt.savefig("Res_and_StimTri-Seaborn-0204-2_1.png", dpi=180)


# In[ ]:




# In[38]:

plt.rcParams['font.family'] = 'Arial'
import matplotlib.font_manager as fm
j_prop = fm.FontProperties(fname='./SourceHanSansJP-Medium.otf', size=18)

fig = plt.figure(figsize=(8,6))
plt.subplots_adjust(hspace=0.1)

ax1=plt.subplot(211)
ax1.plot(data["time[s]"][:80]*10**6, data["Response[V]"][:80]*10**3, label="誘導起電力", color=sns.xkcd_rgb["denim blue"])
ax1.set_ylabel("Voltage[mV]",  fontsize=18)
ax1.set_yticks([-8,-4,0,4,8])
ax1.set_yticklabels([-8,-4,0,4,8], fontsize=16)
ax1.grid(linewidth=1)
ax1.legend(prop=j_prop)
plt.setp(ax1.get_xticklabels(), visible=False)

ax2=plt.subplot(212, sharex=ax1)
ax2.plot(data["time[s]"][:80]*10**6, data["Trigger[V]"][:80], label="Trigger", color=sns.xkcd_rgb["pale red"])
ax2.set_xlabel("time[μs]",  fontsize=18)
ax2.set_ylabel("Voltage[V]",  fontsize=18)
ax2.set_xticklabels([0,50,100,150,200,250,300,350,400], fontsize=16)
ax2.set_yticks([0,0.4,0.8,1.2])
ax2.set_yticklabels([0,0.4,0.8,1.2],fontsize=16)
ax2.grid(linewidth=1)
ax2.legend(fontsize=18)

plt.tight_layout()

#plt.savefig("Res_and_StimTri-Seaborn-0204-NonStim.png", dpi=180)


# In[ ]:



