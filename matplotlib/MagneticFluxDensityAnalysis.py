
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')


# In[2]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as shape
import matplotlib.cm as cm


# In[3]:

data = pd.read_csv("out-z-160102_3V_up_1000.csv", index_col=0)
data_Br = pd.read_csv("out-r-160102_3V_up_z1000_x1000.csv", index_col=0)


# In[9]:

data


# In[9]:

plt.figure(figsize=(12,9))
plt.imshow(data, interpolation="none")
plt.title("Estimation of magnetic flux density", fontsize=28)
plt.xticks([0,10,20,30,40,50,60,70,80], [-1000,-750,-500,-250,0,250,500,750,1000], rotation=45, fontsize=28)
plt.xlabel("[um]", fontsize=28)
plt.yticks([0,20,40,60,80], [-1000,-500,0,500,1000], fontsize=28)
plt.ylabel("[um]", fontsize=28)
cbar = plt.colorbar(orientation="vertical")
cbar.set_label("magnetic flux density [mT]", fontsize=28)
for t in cbar.ax.get_yticklabels():
    t.set_fontsize(28)
plt.tight_layout()
#plt.savefig("out-z-160102_3V_up_1000-update-micro.eps")
plt.savefig("out-z-160102_3V_up_1000-update-micro.png", dpi=180)


# In[4]:

fig=plt.figure(figsize=(12,9))
ax = fig.add_subplot(111)
image=ax.imshow(data, interpolation="none", cmap=cm.viridis)

#Coil
rect1 = shape.Rectangle((31,28), 18, 24, edgecolor="white", facecolor="none", linewidth=5)
ax.add_patch(rect1)
#stage
rect2 = shape.Rectangle((31,20), 18, 8, edgecolor="white", facecolor="none", linewidth=5)
ax.add_patch(rect2)
rect3 = shape.Rectangle((31,52), 18, 8, edgecolor="white", facecolor="none", linewidth=5)
ax.add_patch(rect3)

ax.set_xticks([0,10,20,30,40,50,60,70,80])
ax.set_xticklabels([-1000,-750,-500,-250,0,250,500,750,1000], fontsize=28, rotation=45)
ax.set_xlabel("[um]", fontsize=28)

ax.set_yticks([0,20,40,60,80])
ax.set_yticklabels([-1000,-500,0,500,1000], fontsize=28)
ax.set_ylabel("[um]", fontsize=28)

ax.set_title("Estimation of magnetic flux density", fontsize=28)

cbar=plt.colorbar(image)
cbar.set_label("magnetic flux density[mT]", fontsize=28)
for t in cbar.ax.get_yticklabels():
    t.set_fontsize(28)
    
plt.tight_layout()
#plt.savefig("out-z-160102_3V_up_1000-update-rect.eps")
plt.savefig("out-z-160102_3V_up_1000-update-rect-viridis.png", dpi=180)


# In[18]:

fig=plt.figure(figsize=(12,9))
ax = fig.add_subplot(111)
image=ax.imshow(data, interpolation="none", cmap=cm.jet)

#Coil
rect1 = shape.Rectangle((31,28), 18, 24, edgecolor="white", facecolor="none", linewidth=5)
ax.add_patch(rect1)
#stage
rect2 = shape.Rectangle((31,20), 18, 8, edgecolor="white", facecolor="none", linewidth=5)
ax.add_patch(rect2)
rect3 = shape.Rectangle((31,52), 18, 8, edgecolor="white", facecolor="none", linewidth=5)
ax.add_patch(rect3)

ax.set_xticks([0,10,20,30,40,50,60,70,80])
ax.set_xticklabels([-1000,-750,-500,-250,0,250,500,750,1000], fontsize=28, rotation=45)
ax.set_xlabel("[µm]", fontsize=28)

ax.set_yticks([0,20,40,60,80])
ax.set_yticklabels([-1000,-500,0,500,1000], fontsize=28)
ax.set_ylabel("[µm]", fontsize=28)

ax.set_title("Estimation of Bz", fontsize=28)

cbar=plt.colorbar(image)
cbar.set_label("magnetic flux density [mT]", fontsize=28)
for t in cbar.ax.get_yticklabels():
    t.set_fontsize(28)
    
plt.tight_layout()
plt.savefig("out-z-160102_3V_up_1000-update-rect-micro-Bz.eps")
#plt.savefig("out-z-160102_3V_up_1000-update-rect-micro-Bz.png", dpi=180)


# In[7]:

fig=plt.figure(figsize=(12,9))
ax = fig.add_subplot(111)
image=ax.imshow(data_Br, interpolation="none", cmap=cm.jet)

#Coil
rect1 = shape.Rectangle((31,28), 18, 24, edgecolor="white", facecolor="none", linewidth=5)
ax.add_patch(rect1)
#stage
rect2 = shape.Rectangle((31,20), 18, 8, edgecolor="white", facecolor="none", linewidth=5)
ax.add_patch(rect2)
rect3 = shape.Rectangle((31,52), 18, 8, edgecolor="white", facecolor="none", linewidth=5)
ax.add_patch(rect3)

ax.set_xticks([0,10,20,30,40,50,60,70,80])
ax.set_xticklabels([-1000,-750,-500,-250,0,250,500,750,1000], fontsize=28, rotation=45)
ax.set_xlabel("[µm]", fontsize=28)

ax.set_yticks([0,20,40,60,80])
ax.set_yticklabels([-1000,-500,0,500,1000], fontsize=28)
ax.set_ylabel("[µm]", fontsize=28)

ax.set_title("Estimation of Bx", fontsize=28)

cbar=plt.colorbar(image)
cbar.set_label("magnetic flux density [mT]", fontsize=28)
for t in cbar.ax.get_yticklabels():
    t.set_fontsize(28)
    
plt.tight_layout()
#plt.savefig("out-z-160102_3V_up_1000-update-rect-micro.eps")
plt.savefig("out-r-160102_3V_up_1000-update-rect-micro.png", dpi=180)


# In[17]:

fig=plt.figure(figsize=(12,9))
ax = fig.add_subplot(111)
image=ax.imshow(data, interpolation="none", cmap=cm.jet)

#Coil
rect1 = shape.Rectangle((31,28), 18, 24, edgecolor="white", facecolor="none", linewidth=5)
ax.add_patch(rect1)
#stage
rect2 = shape.Rectangle((31,20), 18, 8, edgecolor="white", facecolor="none", linewidth=5)
ax.add_patch(rect2)
rect3 = shape.Rectangle((31,52), 18, 8, edgecolor="white", facecolor="none", linewidth=5)
ax.add_patch(rect3)

ax.set_xticks([0,10,20,30,40,50,60,70,80])
ax.set_xticklabels([-1000,-750,-500,-250,0,250,500,750,1000], fontsize=28, rotation=45)
ax.set_xlabel("[µm]", fontsize=28)

ax.set_yticks([0,20,40,60,80])
ax.set_yticklabels([-1000,-500,0,500,1000], fontsize=28)
ax.set_ylabel("[µm]", fontsize=28)

ax.set_title(r"Estimation of $B_{z}$",fontsize=28)

cbar=plt.colorbar(image)
cbar.set_label("magnetic flux density [mT]", fontsize=28)
for t in cbar.ax.get_yticklabels():
    t.set_fontsize(28)
    
plt.tight_layout()
#plt.savefig("out-z-160102_3V_up_1000-update-rect-micro-z.eps")
plt.savefig("out-z-160102_3V_up_1000-update-rect-micro-z.png", dpi=180)


# In[ ]:



