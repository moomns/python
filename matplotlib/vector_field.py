
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')


# In[23]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as shape


# In[13]:

U = pd.read_csv("out-r-160102_3V_up_z1000_x1000.csv", index_col=0)
V = pd.read_csv("out-z-160102_3V_up_z1000_x1000.csv", index_col=0)


# In[17]:

x = np.arange(-1000, 1025, 25)
z = np.arange(1000, -1025, -25)
X,Z=np.meshgrid(x,z)


# In[22]:

plt.figure(figsize=(36,24))
plt.quiver(X,Z,U*10,V*10)


# In[ ]:

"fig=plt.figure(figsize=(12,9))\n",
"ax = fig.add_subplot(111)\n",
"image=ax.imshow(data, interpolation=\"none\", cmap=cm.viridis)\n",
"\n",
"#Coil\n",
"rect1 = shape.Rectangle((31,28), 18, 24, edgecolor=\"white\", facecolor=\"none\", linewidth=5)\n",
"ax.add_patch(rect1)\n",
"#stage\n",
"rect2 = shape.Rectangle((31,20), 18, 8, edgecolor=\"white\", facecolor=\"none\", linewidth=5)\n",
"ax.add_patch(rect2)\n",
"rect3 = shape.Rectangle((31,52), 18, 8, edgecolor=\"white\", facecolor=\"none\", linewidth=5)\n",
"ax.add_patch(rect3)\n",
"\n",
"ax.set_xticks([0,10,20,30,40,50,60,70,80])\n",
"ax.set_xticklabels([-1000,-750,-500,-250,0,250,500,750,1000], fontsize=28, rotation=45)\n",
"ax.set_xlabel(\"[um]\", fontsize=28)\n",
"\n",
"ax.set_yticks([0,20,40,60,80])\n",
"ax.set_yticklabels([-1000,-500,0,500,1000], fontsize=28)\n",
"ax.set_ylabel(\"[um]\", fontsize=28)\n",
"\n",
"ax.set_title(\"Estimation of magnetic flux density\", fontsize=28)\n",
"\n",
"cbar=plt.colorbar(image)\n",
"cbar.set_label(\"magnetic flux density[mT]\", fontsize=28)\n",
"for t in cbar.ax.get_yticklabels():\n",
"    t.set_fontsize(28)\n",
"    \n",
"plt.tight_layout()\n",


# In[48]:

fig = plt.figure(figsize=(8,8))
plt.rcParams['font.family']= 'Times New Roman'
ax = fig.add_subplot(111)
ax.quiver(X,Z,U,V)
rect1 = shape.Rectangle((-200,-300), 400, 600, edgecolor="blue", facecolor="none", linewidth=5, alpha=0.5)
ax.add_patch(rect1)
rect2 = shape.Rectangle((-200,300), 400, 200, edgecolor="blue", facecolor="none", linewidth=5, alpha=0.5)
ax.add_patch(rect2)
rect3 = shape.Rectangle((-200,-500), 400, 200, edgecolor="blue", facecolor="none", linewidth=5, alpha=0.5)
ax.add_patch(rect3)

ax.set_xlabel("[μm]", fontsize=28)
ax.set_ylabel("[μm]", fontsize=28)
ax.set_xticklabels([-1000,-500,0,500,1000], fontsize=28, rotation=45)
ax.set_yticklabels([-1000,-500,0,500,1000], fontsize=28)
plt.savefig("magnetic_flux_density_vector_field.png", dpi=180)


# In[ ]:



