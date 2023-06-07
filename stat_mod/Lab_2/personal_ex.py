import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, chi2

lam = 4
n = 100
sample = np.random.poisson(lam, n)

plt.hist(sample, bins=10, density=True)
plt.title('Гистограмма распределения')
plt.show()

observed, bins = np.histogram(sample, bins=10)
print("Эмпирическая: ", observed)
expected = np.array([len(sample) * (poisson.cdf(bins[i+1], lam) - poisson.cdf(bins[i], lam)) for i in range(10)])
print("Теоретическая: ", expected)

hi2 = np.sum((observed - expected) ** 2 / expected)
critical_value = chi2.isf(0.05, 9, loc=0, scale=1)

print('Значение хи-квадрат:', hi2)
print("критическое значение:", critical_value)
