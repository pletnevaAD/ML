import math

import numpy as np
from prettytable import PrettyTable

from Lab_2.get_plots import get_plots


def uniform_dist(n, r_low, r_up):
    r_array = []
    D = 0
    for i in range(n):
        r = math.floor((r_up - r_low + 1) * np.random.random() + r_low)
        r_array.append(r)
    M = sum(r_array)/n
    for i in range(n):
        D += (r_array[i] - M) ** 2
    D /= n
    return r_array, M, D


n = 10000
r_low = 1
r_up = 100
array, M, D = uniform_dist(n, r_low, r_up)
M_theor = (r_low + r_up) / 2
D_theor = ((r_up - r_low + 1) ** 2 - 1) / 12
table = PrettyTable(['Оценка распределений', 'Эксперимент', 'Теоретическое значение', 'Отклонение'])
table.add_row(["M", M, M_theor, abs(M - M_theor)])
table.add_row(["D", D, D_theor, abs(D - D_theor)])
print(table)

get_plots(array)
