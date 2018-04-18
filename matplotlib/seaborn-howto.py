# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("darkgrid")
start = 400
end = 650
plt.plot(ampout.time[start:end]*10**6, ampout.data[start:end])
plt.xlabel("time [μs]", fontsize=18)
plt.xticks(fontsize=18)
plt.ylabel("Voltage [V]", fontsize=18)
plt.yticks(fontsize=18)
plt.title("0-3 V square wave -> Amp ", fontsize=18)
plt.grid(linewidth=2, alpha=0.8)
absmax = np.max(ampout.data[start:end])
argmax = np.argmax(ampout.data[start:end])
plt.annotate("Max:{0:.3f} [V]".format(absmax), xy=(ampout.time[argmax]*10**6, absmax), xytext=(ampout.time[argmax+50]*10**6, absmax-0.1), arrowprops=dict(arrowstyle="->",connectionstyle="arc3", linewidth=2), fontsize=18)
absmin = np.min(ampout.data[argmax-20:argmax])
argmin = np.argmin(ampout.data[argmax-20:argmax])
absmin = ampout.data[argmin]
plt.annotate("Min:{0:.3f} [V]".format(absmin), xy=(ampout.time[argmin]*10**6, absmin), xytext=(ampout.time[argmin+50]*10**6, absmin), arrowprops=dict(arrowstyle="->",connectionstyle="arc3", linewidth=2), fontsize=18)
plt.annotate("Vpp={} [V]".format(absmax-absmin), xy=(105,4), xytext=(105,4), size=18)
plt.tight_layout()
plt.savefig("MaxCurrentCalculate-Seaborn.png", dpi=180)


##
# sns.pairplotのいじり方
#http://statmodeling.hatenablog.com/entry/pystan-rstanbook-chap5-1
#http://sinhrks.hatenablog.com/
#https://stackoverflow.com/questions/30942577/seaborn-correlation-coefficient-on-pairgrid
##

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