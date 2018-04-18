
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')


# In[31]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as shape


# In[19]:

data = pd.read_csv("tmp.csv", index_col="index")
plt.figure(figsize=(12,8))
plt.imshow(data, interpolation="none")
plt.xticks([0,10,20,30,40,50,60], [-750,-500,-250,0,250,500,750], fontsize=28, rotation=45)
plt.xlabel("[um]", fontsize=28)
plt.yticks([0,20,40,60,80], [-1000,-500,0,500,1000], fontsize=28)
plt.ylabel("[um]", fontsize=28)
plt.title("Estimation", fontsize=28)
cbar=plt.colorbar()
cbar.set_label("magnetic flux density[mT]", fontsize=28)
for t in cbar.ax.get_yticklabels():
    t.set_fontsize(28)
plt.tight_layout()


# In[108]:

data = pd.read_csv("tmp.csv", index_col="index")
fig=plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
image=ax.imshow(data, interpolation="none")

#Coil
rect1 = shape.Rectangle((21,28), 18, 24, edgecolor="white", facecolor="none", linewidth=5)
ax.add_patch(rect1)
#stage
rect2 = shape.Rectangle((21,20), 18, 8, edgecolor="white", facecolor="none", linewidth=5)
ax.add_patch(rect2)
rect3 = shape.Rectangle((21,52), 18, 8, edgecolor="white", facecolor="none", linewidth=5)
ax.add_patch(rect3)


ax.set_xticks([0,10,20,30,40,50,60])
ax.set_xticklabels([-750,-500,-250,0,250,500,750], fontsize=28, rotation=45)
ax.set_xlabel("[um]", fontsize=28)

ax.set_yticks([0,20,40,60,80])
ax.set_yticklabels([-1000,-500,0,500,1000], fontsize=28)
ax.set_ylabel("[um]", fontsize=28)

ax.set_title("Estimation", fontsize=28)

cbar=plt.colorbar(image)
cbar.set_label("magnetic flux density[mT]", fontsize=28)
for t in cbar.ax.get_yticklabels():
    t.set_fontsize(28)
    
plt.tight_layout()


# In[65]:

data.ix[100:]


# In[ ]:



