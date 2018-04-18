
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')


# In[2]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("poster")


# In[3]:

anawave1 = pd.read_csv("1159-5kHz-ExtractROI-anawave.csv", skiprows=10)
anawave1 = anawave1.ix[:, :3]
anawave1.columns=["time (ms)", "raw", "change rate (%)"]


# In[4]:

centerwave1 = pd.read_csv("1159-5kHz-ExtractROI-wave-center.csv", skiprows=10)
centerwave1 = centerwave1.ix[:, :3]
centerwave1.columns=["time (ms)", "raw", "change rate (%)"]
centerwave1["inverse (%)"] = centerwave1["change rate (%)"] * -1


# In[5]:

rightwave1 = pd.read_csv("1159-5kHz-ExtractROI-wave-right.csv", skiprows=10)
rightwave1 = rightwave1.ix[:, :3]
rightwave1.columns=["time (ms)", "raw", "change rate (%)"]
rightwave1["inverse (%)"] = rightwave1["change rate (%)"] * -1


# In[13]:

fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(311)
ax1.plot(anawave1["time (ms)"], anawave1["change rate (%)"], label="Stim")
plt.setp(ax1.get_yticklabels(), visible=False)
ax1.legend()
ax2 = fig.add_subplot(312, sharex=ax1)
plt.axhline(y=0, color="white")
ax2.plot(centerwave1["time (ms)"], centerwave1["inverse (%)"], label="center")
ax2.set_ylabel("(%)")
ax2.set_yticks([-0.4,0,0.4])
ax2.set_yticklabels([-0.4,0,0.4])
ax2.legend()
ax3 = fig.add_subplot(313, sharex=ax1)
plt.axhline(y=0, color="white")
ax3.plot(rightwave1["time (ms)"], rightwave1["inverse (%)"], label="right")
ax3.set_xlabel("time (ms)")
ax3.set_ylabel("(%)")
ax3.set_yticks([-0.4,0,0.4])
ax3.set_yticklabels([-0.4,0,0.4])
ax3.legend()

xticklabels=ax1.get_xticklabels()+ax2.get_xticklabels()
plt.setp(xticklabels, visible=False)

plt.savefig("1159-5kHz-ExtractROI-wave.eps")


# In[8]:

anawave2 = pd.read_csv("1245-MultiA5-ExtractROI-anawave.csv", skiprows=10)
anawave2 = anawave2.ix[:, :3]
anawave2.columns=["time (ms)", "raw", "change rate (%)"]


# In[9]:

centerwave2 = pd.read_csv("1245-MultiA5-ExtractROI-wave-center.csv", skiprows=10)
centerwave2 = centerwave2.ix[:, :3]
centerwave2.columns=["time (ms)", "raw", "change rate (%)"]
centerwave2["inverse (%)"] = centerwave2["change rate (%)"] * -1


# In[10]:

rightwave2 = pd.read_csv("1245-MultiA5-ExtractROI-wave-right.csv", skiprows=10)
rightwave2 = rightwave2.ix[:, :3]
rightwave2.columns=["time (ms)", "raw", "change rate (%)"]
rightwave2["inverse (%)"] = rightwave2["change rate (%)"] * -1


# In[11]:

leftwave2 = pd.read_csv("1245-MultiA5-ExtractROI-wave-left.csv", skiprows=10)
leftwave2 = leftwave2.ix[:, :3]
leftwave2.columns=["time (ms)", "raw", "change rate (%)"]
leftwave2["inverse (%)"] = leftwave2["change rate (%)"] * -1


# In[14]:

fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(311)
ax1.plot(anawave2["time (ms)"], anawave2["change rate (%)"], label="Stim")
plt.setp(ax1.get_yticklabels(), visible=False)
ax1.legend()
ax2 = fig.add_subplot(312, sharex=ax1)
plt.axhline(y=0, color="white")
ax2.plot(centerwave2["time (ms)"], centerwave2["inverse (%)"], label="center")
ax2.set_ylabel("(%)")
#ax2.set_yticks([-0.4,0,0.4])
#ax2.set_yticklabels([-0.4,0,0.4])
ax2.legend()
ax3 = fig.add_subplot(313, sharex=ax1)
plt.axhline(y=0, color="white")
ax3.plot(rightwave2["time (ms)"], rightwave2["inverse (%)"], label="right")
ax3.set_xlabel("time (ms)")
ax3.set_ylabel("(%)")
#ax3.set_yticks([-0.4,0,0.4])
#ax3.set_yticklabels([-0.4,0,0.4])
ax3.legend(loc=4)

xticklabels=ax1.get_xticklabels()+ax2.get_xticklabels()
plt.setp(xticklabels, visible=False)

plt.savefig("1245-MultiA5-ExtractROI-wave.eps")


# In[81]:

anawave3 = pd.read_csv("1322-MultiA6-ExtractROI-anawave.csv", skiprows=10)
anawave3 = anawave3.ix[:, :3]
anawave3.columns=["time (ms)", "raw", "change rate (%)"]


# In[85]:

rightwave3 = pd.read_csv("1322-MultiA6-ExtractROI-wave-right.csv", skiprows=10)
rightwave3 = rightwave3.ix[:, :3]
rightwave3.columns=["time (ms)", "raw", "change rate (%)"]
rightwave3["inverse (%)"] = rightwave3["change rate (%)"] * -1


# In[87]:

centerwave3 = pd.read_csv("1322-MultiA6-ExtractROI-wave-center.csv", skiprows=10)
centerwave3 = centerwave3.ix[:,3]
centerwave3.columns=["change rate (%)"]
centerwave3
#centerwave3["inverse (%)"] = centerwave3["change rate (%)"] * -1


# In[89]:

leftwave3 = pd.read_csv("1322-MultiA6-ExtractROI-wave-left.csv", skiprows=10)
leftwave3 = leftwave3.ix[:, 4]


# In[100]:

fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(311)
ax1.plot(anawave3["time (ms)"], anawave3["change rate (%)"], label="Stim")
plt.setp(ax1.get_yticklabels(), visible=False)
ax1.legend()
ax2 = fig.add_subplot(312, sharex=ax1)
ax2.plot(rightwave3["time (ms)"], centerwave3*(-1), label="center")
ax2.set_ylabel("(%)")
#ax2.set_yticks([-0.4,0,0.4])
#ax2.set_yticklabels([-0.4,0,0.4])
ax2.legend()
ax3 = fig.add_subplot(313, sharex=ax1)
ax3.plot(rightwave3["time (ms)"], rightwave3["inverse (%)"], label="right")
ax3.set_xlabel("time (ms)")
ax3.set_ylabel("(%)")
#ax3.set_yticks([-0.4,0,0.4])
#ax3.set_yticklabels([-0.4,0,0.4])
ax3.legend(loc=4)

xticklabels=ax1.get_xticklabels()+ax2.get_xticklabels()
plt.setp(xticklabels, visible=False)

plt.savefig("1322-MultiA6-ExtractROI-wave.eps")


# In[90]:

anawave4 = pd.read_csv("1322-MultiA7-ExtractROI-anawave.csv", skiprows=10)
anawave4 = anawave4.ix[:, :3]
anawave4.columns=["time (ms)", "raw", "change rate (%)"]


# In[91]:

rightwave4 = pd.read_csv("1322-MultiA7-ExtractROI-wave-right.csv", skiprows=10)
rightwave4 = rightwave3.ix[:, :3]
rightwave4.columns=["time (ms)", "raw", "change rate (%)"]
rightwave4["inverse (%)"] = rightwave3["change rate (%)"] * -1


# In[97]:

leftwave4 = pd.read_csv("1322-MultiA7-ExtractROI-wave-left.csv", skiprows=10)
leftwave4 = leftwave4.ix[:,3]
leftwave4.columns=["change rate (%)"]
centerwave4
#centerwave3["inverse (%)"] = centerwave3["change rate (%)"] * -1


# In[99]:

fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(311)
ax1.plot(anawave4["time (ms)"], anawave4["change rate (%)"], label="Stim")
plt.setp(ax1.get_yticklabels(), visible=False)
ax1.legend()
ax2 = fig.add_subplot(312, sharex=ax1)
ax2.plot(rightwave4["time (ms)"], leftwave4*(-1), label="center")
ax2.set_ylabel("(%)")
#ax2.set_yticks([-0.4,0,0.4])
#ax2.set_yticklabels([-0.4,0,0.4])
ax2.legend()
ax3 = fig.add_subplot(313, sharex=ax1)
ax3.plot(rightwave4["time (ms)"], rightwave4["inverse (%)"], label="right")
ax3.set_xlabel("time (ms)")
ax3.set_ylabel("(%)")
#ax3.set_yticks([-0.4,0,0.4])
#ax3.set_yticklabels([-0.4,0,0.4])
ax3.legend(loc=4)

xticklabels=ax1.get_xticklabels()+ax2.get_xticklabels()
plt.setp(xticklabels, visible=False)

plt.savefig("1322-MultiA7-ExtractROI-wave.eps")


# In[18]:

anawave5 = pd.read_csv("1322-MultiA11-ExtractROI-anawave.csv", skiprows=10)
anawave5 = anawave5.ix[:, :3]
anawave5.columns=["time (ms)", "raw", "change rate (%)"]


# In[19]:

leftwave5 = pd.read_csv("1322-MultiA11-ExtractROI-leftwave.csv", skiprows=10)
leftwave5 = leftwave5.ix[:, :3]
leftwave5.columns=["time (ms)", "raw", "change rate (%)"]
leftwave5["inverse (%)"] = leftwave5["change rate (%)"] * -1


# In[20]:

rightwave5 = pd.read_csv("1322-MultiA11-ExtractROI-leftwave.csv", skiprows=10)
rightwave5 = rightwave5.ix[:, 3]


# In[21]:

fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(311)
ax1.plot(anawave5["time (ms)"], anawave5["change rate (%)"], label="Stim")
plt.setp(ax1.get_yticklabels(), visible=False)
ax1.legend()
ax2 = fig.add_subplot(312, sharex=ax1)
ax2.plot(leftwave5["time (ms)"], leftwave5["inverse (%)"], label="left")
ax2.set_ylabel("(%)")
#ax2.set_yticks([-0.4,0,0.4])
#ax2.set_yticklabels([-0.4,0,0.4])
ax2.legend()
ax3 = fig.add_subplot(313, sharex=ax1)
ax3.plot(leftwave5["time (ms)"], rightwave5*(-1), label="right")
ax3.set_xlabel("time (ms)")
ax3.set_ylabel("(%)")
#ax3.set_yticks([-0.4,0,0.4])
#ax3.set_yticklabels([-0.4,0,0.4])
ax3.legend(loc=4)

xticklabels=ax1.get_xticklabels()+ax2.get_xticklabels()
plt.setp(xticklabels, visible=False)

plt.savefig("1322-MultiA11-ExtractROI-wave.png", dpi=300)


# In[ ]:



