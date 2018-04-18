import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

with open('output/model_and_result.pkl', 'rb') as f:
    stanmodel = pickle.load(f)
    fit_nuts = pickle.load(f)

d_ori = pd.read_csv('input/data-attendance-1.txt')
ms = fit_nuts.extract()

quantile = [10, 50, 90]
colname = ['p' + str(x) for x in quantile]
d_qua = pd.DataFrame(np.percentile(ms['y_pred'], q=quantile, axis=0).T, columns=colname)
d = pd.concat([d_ori, d_qua], axis=1)
d0 = d[d.A == 0]
d1 = d[d.A == 1]

palette = sns.color_palette()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([0,0.5], [0,0.5], 'k--', alpha=0.7)
ax.errorbar(d0.Y, d0.p50, yerr=[d0.p50 - d0.p10, d0.p90 - d0.p50],
    fmt='o', ecolor='gray', ms=10, mfc=palette[0], alpha=0.8, marker='o')
ax.errorbar(d1.Y, d1.p50, yerr=[d1.p50 - d1.p10, d1.p90 - d1.p50],
    fmt='o', ecolor='gray', ms=10, mfc=palette[1], alpha=0.8, marker='^')
ax.set_aspect('equal')
ax.set_xlim(0, 0.5)
ax.set_ylim(0, 0.5)
ax.set_xlabel('Observed')
ax.set_ylabel('Predicted')
fig.savefig('output/fig5-3.png')