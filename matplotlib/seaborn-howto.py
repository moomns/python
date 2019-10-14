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

"""
対角プロット
下三角にはKDEを
上三角にはスピアマンの順位相関係数を表示
"""
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
g = g.map_diag(sns.distplot, kde=True) #kde=False
g = g.map_upper(corrfunc)
g.fig.subplots_adjust(wspace=0.05, hspace=0.05)
for ax in g.axes.flatten():
    for t in ax.get_xticklabels():
        _ = t.set(rotation=40)
g.savefig('output/fig5-5.png')

"""
対角プロット
ピアソンの積率相関係数とそのp値を表示
"""
def corrfunc(x, y, **kws):
    (r, p) = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f} ".format(r),
                xy=(.1, .9), xycoords=ax.transAxes, fontsize=12)
    ax.annotate("p = {:.3f}".format(p),
                xy=(.1, .8), xycoords=ax.transAxes, fontsize=12)

df = sns.load_dataset("iris")
df = df[df["species"] == "setosa"]
graph = sns.pairplot(df)
graph.map(corrfunc)
plt.show()



"""
散布図とヒストグラムと相関行列
"""

def hist_scatter_cor_matrix(dataset, hue=None, filename=False):
    from scipy import stats
    import math
    from matplotlib.patches import Ellipse
    import seaborn as sns

    def corrfunc(x, y, **kws):
        #Spearmanの順位相関係数
        #母集団が正規分布に乗らない場合や、順位情報しかない場合に用いる
        #尺度水準が比率、感覚、順序尺度のデータを用いること脱できる
        r, _ = stats.spearmanr(x, y)
        ax = plt.gca()
        ax.axis('off')
        ellcolor = plt.cm.coolwarm(0.5*(r+1))#cm.RdBu
        txtcolor = 'black' if math.fabs(r) < 0.5 else 'white'
        ax.add_artist(Ellipse(xy=[.5, .5], width=math.sqrt(1+r), height=math.sqrt(1-r), angle=45,
            facecolor=ellcolor, edgecolor='none', transform=ax.transAxes))
        ax.text(.5, .5, '{:.0f}'.format(r*100), color=txtcolor, fontsize=28,
            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    plt.figure(figsize=(10,10))
    g = sns.PairGrid(dataset, hue=hue)
    g = g.map_lower(plt.scatter, cmap='Blues_d')
    g = g.map_diag(sns.distplot, kde=False) #kde=False
    g = g.map_upper(corrfunc)
    g.fig.subplots_adjust(wspace=0.05, hspace=0.05)
    for ax in g.axes.flatten():
        for t in ax.get_xticklabels():
            _ = t.set(rotation=40)
    plt.tight_layout()

    if filename:
        plt.savefig(filename+".png", dpi=300)
        plt.savefig(filename+".pdf")


"""
seaborn.JointGrid.annotateから改変
既存の図に統計量とそのp値を表示する

frameon=False
"""
def annotate(ax, x, y, func, template=None, stat=None, loc="best", **kwargs):
    """Annotate the plot with a statistic about the relationship.

    Parameters
    ----------
    func : callable
        Statistical function that maps the x, y vectors either to (val, p)
        or to val.
    template : string format template, optional
        The template must have the format keys "stat" and "val";
        if `func` returns a p value, it should also have the key "p".
    stat : string, optional
        Name to use for the statistic in the annotation, by default it
        uses the name of `func`.
        scipy.stats.pearsonr
        scipy.stats.speamanr
    loc : string or int, optional
        Matplotlib legend location code; used to place the annotation.
    kwargs : key, value mappings
        Other keyword arguments are passed to `ax.legend`, which formats
        the annotation.
        legendの枠を消すならframeon=False

    Returns
    -------
    self : JointGrid instance.
        Returns `self`.

    """
    default_template = "{stat} = {val:.2g}; p = {p:.2g}"

    # Call the function and determine the form of the return value(s)
    out = func(x, y)
    try:
        val, p = out
    except TypeError:
        val, p = out, None
        default_template, _ = default_template.split(";")

    # Set the default template
    if template is None:
        template = default_template

    # Default to name of the function
    if stat is None:
        stat = func.__name__

    # Format the annotation
    if p is None:
        annotation = template.format(stat=stat, val=val)
    else:
        annotation = template.format(stat=stat, val=val, p=p)

    # Draw an invisible plot and use the legend to draw the annotation
    # This is a bit of a hack, but `loc=best` works nicely and is not
    # easily abstracted.
    phantom, = ax.plot.plot(x, y, linestyle="", alpha=0)
    ax.legend([phantom], [annotation], loc=loc, **kwargs)
    phantom.remove()

    return ax


#有意差のプロットの仕方
#https://stackoverflow.com/questions/36578458/how-does-one-insert-statistical-annotations-stars-or-p-values-into-matplotlib
#(A)
import seaborn as sns, matplotlib.pyplot as plt

tips = sns.load_dataset("tips")
sns.boxplot(x="day", y="total_bill", data=tips, palette="PRGn")

# statistical annotation
x1, x2 = 2, 3   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
y, h, col = tips['total_bill'].max() + 2, 2, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "ns", ha='center', va='bottom', color=col)

plt.show()

#(B)
#statannot
#https://github.com/webermarcolivier/statannot
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation

sns.set(style="whitegrid")
df = sns.load_dataset("tips")

x = "day"
y = "total_bill"
order = ['Sun', 'Thur', 'Fri', 'Sat']
ax = sns.boxplot(data=df, x=x, y=y, order=order)
add_stat_annotation(ax, data=df, x=x, y=y, order=order,
                    box_pairs=[("Thur", "Fri"), ("Thur", "Sat"), ("Fri", "Sun")],
                    test='Mann-Whitney', text_format='star', loc='outside', verbose=2)

#(C)
x = "day"
y = "total_bill"
hue = "smoker"
ax = sns.boxplot(data=df, x=x, y=y, hue=hue)
add_stat_annotation(ax, data=df, x=x, y=y, hue=hue,
                    box_pairs=[(("Thur", "No"), ("Fri", "No")),
                                 (("Sat", "Yes"), ("Sat", "No")),
                                 (("Sun", "No"), ("Thur", "Yes"))
                                ],
                    test='t-test_ind', text_format='full', loc='inside', verbose=2)
plt.legend(loc='upper left', bbox_to_anchor=(1.03, 1))


##
# データの分布を見るための様々なプロット
##

#hueごとの1次元散布図を一直線に描くｰ>観測値の多い箇所は濃くなる
sns.stripplot(x=x, y=y)
#hueごとの散布図を、各観測点が被さらないように描く
sns.swarmplot(x=x, y=y)
#通常の箱ひげ図。外れ値は、第1/3四分位ー中央値＋第1/3四分位ー中央値の1.5倍を超えた箇所
sns.boxplot(x=x, y=y)
#バイオリンプロット。swarmplotをkdeで描く
sns.violinplot(x=x, y=y)
