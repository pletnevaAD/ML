import numpy as np
import scipy
from prettytable import PrettyTable
import matplotlib.pyplot as plt


def calc_m_d(n_array, rand_array, table):
    fig, axes = plt.subplots(nrows=4)
    plt.subplots_adjust(hspace=0.8)
    plt.figure(1)
    for index, n in enumerate(n_array):
        M = 0
        D = 0
        K = [0] * n
        for i in range(n):
            M += rand_array[i]
        M /= n
        for i in range(n):
            D += (rand_array[i] - M) ** 2
        D /= n
        for f in range(n):
            K[f] = 0
            for i in range(n - f):
                K[f] += (rand_array[i] - M) * (rand_array[i + f] - M)
            K[f] /= D * n

        axes[index].bar(np.arange(1, n),
                        np.array(K[1:n])
                        )
        axes[index].set_title(f'Корреллограмма при n={n}')
        table.add_row([n, 'M', M, 0.5, abs(0.5 - M)])
        table.add_row([n, 'D', D, 0.08333, abs(0.08333 - D)])


table = PrettyTable(['n', 'Оценка распределений', 'RAND (эксперимент)', 'Теоретическое значение', 'Отклонение'])
n = [10, 100, 1000, 10000]
rand_array = np.random.random(10000)
calc_m_d(n, rand_array, table)
data_sorted = np.sort(rand_array)
p = 1. * np.arange(len(rand_array)) / (len(rand_array) - 1)
plt.figure(2)
fig, axes = plt.subplots()
axes.plot(data_sorted, p)
axes.set_title(f'Функция распределения')
plt.figure(3)
figure, ax = plt.subplots()
ax.hist(rand_array, density=True)
ax.set_title(f'Плотность распределения')
plt.show()
print(table)
