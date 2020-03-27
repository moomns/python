from scipy.stats import norm
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

def inductance_fit(x, a, b):
    return a * np.exp(-x/b)

plt.figure(figsize=(10, 5))

plt.scatter(x, y, marker="v")
popt, pcov = curve_fit(inductance_fit, x, y)
plt.plot(x, inductance_fit(x, *popt))

#https://omedstu.jimdofree.com/2018/07/16/python%E3%81%A7gaussian-fitting/
# estimate standard Error
SE = np.sqrt(np.diag(pcov))

# estimate 95% confidence interval
significance_level=0.05
alpha=significance_level/2
#1.96 * SD/sqrt(n) = 1.96 * SE in 95% confidence interval
lwCI = popt + norm.ppf(q=alpha)*SE
upCI = popt + norm.ppf(q=1-alpha)*SE


plt.fill_between(x, inductance_fit(x, *lwCI)), inductance_fit(x, *upCI)), color="gray", alpha=0.25)
plt.legend()


# print result
mat = np.vstack((popt,SE, lwCI, upCI)).T
df=pd.DataFrame(mat,index=("a", "b"),
                columns=("Estimate", "Std. Error", "lwCI","upCI"))
print(df)
