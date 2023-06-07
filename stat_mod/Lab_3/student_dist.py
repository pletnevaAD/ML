import math

import numpy as np
from prettytable import PrettyTable

from Lab_2.get_plots import get_plots


def uniform_dist(n, N):
    r_array = []
    D = 0
    for i in range(n):
        YN = 0
        for i in range(N):
            a = np.random.random(2)
            m = math.sqrt(-2 * math.log(a[1])) * math.cos(2 * math.pi * a[0])
            YN += m ** 2
        a = np.random.random(2)
        m = math.sqrt(-2 * math.log(a[1])) * math.cos(2 * math.pi * a[0])
        r = m / math.sqrt(YN / N)
        r_array.append(r)
    M = sum(r_array) / n
    for i in range(n):
        D += (r_array[i] - M) ** 2
    D /= n
    return r_array, M, D


n = 10000
N = 10
array, M, D = uniform_dist(n, N)
M_theor = 0
D_theor = 1.25
table = PrettyTable(['Оценка распределений', 'Эксперимент', 'Теоретическое значение', 'Отклонение'])
table.add_row(["M", M, M_theor, abs(M - M_theor)])
table.add_row(["D", D, D_theor, abs(D - D_theor)])
print(table)

get_plots(array)
