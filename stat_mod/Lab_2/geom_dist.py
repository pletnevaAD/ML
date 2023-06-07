import math

import numpy as np
from prettytable import PrettyTable

from Lab_2.get_plots import get_plots


def geom_dist_alg_1(n, p):
    r_array = []
    D = 0
    for i in range(n):
        a = np.random.random()
        p_r = p
        m = 0
        while (a - p_r) >= 0:
            a -= p_r
            p_r *= (1 - p)
            m += 1
        r_array.append(m)
    M = sum(r_array) / n
    for i in range(n):
        D += (r_array[i] - M) ** 2
    D /= n
    return r_array, M, D


def geom_dist_alg_2(n, p):
    r_array = []
    D = 0
    for i in range(n):
        a = np.random.random()
        m = 0
        while a > p:
            a = np.random.random()
            m += 1
        r_array.append(m)
    M = sum(r_array) / n
    for i in range(n):
        D += (r_array[i] - M) ** 2
    D /= n
    return r_array, M, D


def geom_dist_alg_3(n, p):
    r_array = []
    D = 0
    for i in range(n):
        a = np.random.random()
        m = int(math.log(a) / math.log(1 - p)) + 1
        r_array.append(m)
    M = sum(r_array) / n
    for i in range(n):
        D += (r_array[i] - M) ** 2
    D /= n
    return r_array, M, D


n = 10000
p = 0.5
array_1, M_1, D_1 = geom_dist_alg_1(n, p)
array_2, M_2, D_2 = geom_dist_alg_2(n, p)
array_3, M_3, D_3 = geom_dist_alg_3(n, p)
M_theor = 2.0
D_theor = 2.0
table = PrettyTable(
    ['Оценка распределений', 'Эксперимент1', 'Эксперимент2', 'Эксперимент3', 'Теоретическое значение', 'Отклонение1',
     'Отклонение2', 'Отклонение3'])
table.add_row(["M", M_1, M_2, M_3, M_theor, abs(M_1 - M_theor), abs(M_2 - M_theor), abs(M_3 - M_theor)])
table.add_row(["D", D_1, D_2, D_3, D_theor, abs(D_1 - D_theor), abs(D_2 - D_theor), abs(D_3 - D_theor)])
print(table)

get_plots(array_1)
get_plots(array_2)
get_plots(array_3)

