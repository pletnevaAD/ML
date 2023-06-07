import math

import numpy as np
from prettytable import PrettyTable

from Lab_2.get_plots import get_plots


def norm_dist_alg_1(n):
    r_array = []
    D = 0
    for i in range(n):
        a = np.random.random(12)
        m = sum(a) - 6
        r_array.append(m)
    M = sum(r_array) / n
    for i in range(n):
        D += (r_array[i] - M) ** 2
    D /= n
    return r_array, M, D


def norm_dist_alg_2(n):
    r_array = []
    D = 0
    for i in range(n):
        a = np.random.random(2)
        m = math.sqrt(-2*math.log(a[1]))*math.cos(2*math.pi*a[0])
        r_array.append(m)
    M = sum(r_array) / n
    for i in range(n):
        D += (r_array[i] - M) ** 2
    D /= n
    return r_array, M, D


n = 10000
array_1, M_1, D_1 = norm_dist_alg_1(n)
array_2, M_2, D_2 = norm_dist_alg_2(n)
M_theor = 0
D_theor = 1
table = PrettyTable(
    ['Оценка распределений', 'Эксперимент1', 'Эксперимент2', 'Теоретическое значение', 'Отклонение1', 'Отклонение2'])
table.add_row(["M", M_1, M_2, M_theor, abs(M_1 - M_theor), abs(M_2 - M_theor)])
table.add_row(["D", D_1, D_2, D_theor, abs(D_1 - D_theor), abs(D_2 - D_theor)])
print(table)

get_plots(array_1)
get_plots(array_2)
