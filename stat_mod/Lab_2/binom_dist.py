import math

import numpy as np
from prettytable import PrettyTable

from Lab_2.get_plots import get_plots


def binomial_dist(n, p, N):
    r_array = []
    D = 0
    for i in range(n):
        a = np.random.random()
        p_r = (1 - p) ** N
        m = 0
        while (a - p_r) >= 0:
            a -= p_r
            p_r *= ((N - m) / (m + 1)) / (p / (1 - p))
            m += 1
        r_array.append(m)
    M = sum(r_array) / n
    for i in range(n):
        D += (r_array[i] - M) ** 2
    D /= n
    return r_array, M, D


n = 10000
p = 0.5
N = 10
array, M, D = binomial_dist(n, p, N)
M_theor = 5
D_theor = 2.5
table = PrettyTable(['Оценка распределений', 'Эксперимент', 'Теоретическое значение', 'Отклонение'])
table.add_row(["M", M, M_theor, abs(M - M_theor)])
table.add_row(["D", D, D_theor, abs(D - D_theor)])
print(table)

get_plots(array)
