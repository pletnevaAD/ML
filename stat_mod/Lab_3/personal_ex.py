import math

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import weibull_min, kstest


def weibull_dist(n, lam, m):
    r_array = []
    for i in range(n):
        r = (lam ** (-1)) * ((-math.log(np.random.random())) ** (1 / m))
        r_array.append(r)
    return r_array


n = 100
lam = 1
m = 4
array = weibull_dist(n, lam, m)

sns.histplot(
    data=array,
    kde=True,
    line_kws={"lw": 3})
plt.show()

x = np.linspace(0, 4, 100)
pdf = weibull_min.pdf(x, c=m, scale=1/lam)
ks_stat = kstest(array, weibull_min.cdf, args=(m, 0, lam))
print(f'Критерий Колмогорова: статистика = {ks_stat.statistic}')

