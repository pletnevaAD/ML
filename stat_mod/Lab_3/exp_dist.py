import math

import numpy as np
from prettytable import PrettyTable

from Lab_2.get_plots import get_plots


def uniform_dist(n, beta):
    r_array = []
    D = 0
    for i in range(n):
        r = -1 * beta * math.log(np.random.random())
        r_array.append(r)
    M = sum(r_array) / n
    for i in range(n):
        D += (r_array[i] - M) ** 2
    D /= n
    return r_array, M, D


n = 10000
beta = 1
array, M, D = uniform_dist(n, beta)
M_theor = 1
D_theor = 1
table = PrettyTable(['Оценка распределений', 'Эксперимент', 'Теоретическое значение', 'Отклонение'])
table.add_row(["M", M, M_theor, abs(M - M_theor)])
table.add_row(["D", D, D_theor, abs(D - D_theor)])
print(table)

get_plots(array)
