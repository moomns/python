import pickle
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

with open('output/model_and_result.pkl', 'rb') as f:
    stanmodel = pickle.load(f)
    fit_nuts = pickle.load(f)

## summary
with open('output/fit-summary.txt', 'w') as f:
    f.write(str(fit_nuts))

## traceplot
palette = sns.color_palette()
ms = fit_nuts.extract(permuted=False, inc_warmup=True)
iter_from = fit_nuts.sim['warmup']
iter_range = np.arange(iter_from, ms.shape[0])
paraname = fit_nuts.sim['fnames_oi']
num_pages = math.ceil(len(paraname)/4)

with PdfPages('output/fit-traceplot.pdf') as pdf:
    for pg in range(num_pages):
        plt.figure()
        for pos in range(4):
            pi = pg*4 + pos
            if pi >= len(paraname): break
            plt.subplot(4, 2, 2*pos+1)
            plt.tight_layout()
            [plt.plot(iter_range + 1, ms[iter_range,ci,pi], color=palette[ci]) for ci in range(ms.shape[1])]
            plt.title(paraname[pi])
            plt.subplot(4, 2, 2*(pos+1))
            plt.tight_layout()
            [sns.kdeplot(ms[iter_range,ci,pi], color=palette[ci]) for ci in range(ms.shape[1])]
            plt.title(paraname[pi])
        pdf.savefig()
        plt.close()