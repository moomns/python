import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from matplotlib.patches import Ellipse
from scipy import stats

with open('output/model_and_result.pkl', 'rb') as f:
    stanmodel = pickle.load(f)
    fit_nuts = pickle.load(f)

d_ori = pd.read_csv('input/data-attendance-1.txt')
ms = fit_nuts.extract()

def corrfunc(x, y, **kws):
    r, _ = stats.spearmanr(x, y)
    ax = plt.gca()
    ax.axis('off')
    ellcolor = plt.cm.RdBu(0.5*(r+1))
    txtcolor = 'black' if math.fabs(r) < 0.5 else 'white'
    ax.add_artist(Ellipse(xy=[.5, .5], width=math.sqrt(1+r), height=math.sqrt(1-r), angle=45,
        facecolor=ellcolor, edgecolor='none', transform=ax.transAxes))
    ax.text(.5, .5, '{:.0f}'.format(r*100), color=txtcolor, fontsize=28,
        horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

d = pd.DataFrame({'b1':ms['b1'], 'b2':ms['b2'], 'b3':ms['b3'], 'sigma':ms['sigma'],
                  'mu1':ms['mu'][:,0], 'mu50':ms['mu'][:,49], 'lp__':ms['lp__']},
                  columns=['b1', 'b2', 'b3', 'sigma', 'mu1', 'mu50', 'lp__'])
sns.set(font_scale=2)
g = sns.PairGrid(d)
g = g.map_lower(sns.kdeplot, cmap='Blues_d')
g = g.map_diag(sns.distplot, kde=False)
g = g.map_upper(corrfunc)
g.fig.subplots_adjust(wspace=0.05, hspace=0.05)
for ax in g.axes.flatten():
    for t in ax.get_xticklabels():
        _ = t.set(rotation=40)
g.savefig('output/fig5-5.png')
