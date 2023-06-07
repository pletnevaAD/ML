import math

import numpy as np
from prettytable import PrettyTable


from Lab_2.get_plots import get_plots


def puasson_dist_alg_1(n, mu):
    r_array = []
    D = 0
    if mu >= 88:
        r_array = np.random.normal(mu, mu, n)
    else:
        for i in range(n):
            a = np.random.random()
            p_r = math.exp(-mu)
            m = 0
            while (a - p_r) >= 0:
                a -= p_r
                m += 1
                p_r *= mu / m
            r_array.append(m)
    M = sum(r_array) / n
    for i in range(n):
        D += (r_array[i] - M) ** 2
    D /= n
    return r_array, M, D


def puasson_dist_alg_2(n, mu):
    r_array = []
    D = 0
    if mu >= 88:
        r_array = np.random.normal(mu, mu, n)
    else:
        for i in range(n):
            a = np.random.random()
            m = 0
            while a >= math.exp(-mu):
                a *= np.random.random()
                m += 1
            r_array.append(m)
    M = sum(r_array) / n
    for i in range(n):
        D += (r_array[i] - M) ** 2
    D /= n
    return r_array, M, D


n = 10000
mu = 10
array_1, M_1, D_1 = puasson_dist_alg_1(n, mu)
array_2, M_2, D_2 = puasson_dist_alg_2(n, mu)
M_theor = 10.0
D_theor = 10.0
table = PrettyTable(
    ['Оценка распределений', 'Эксперимент1', 'Эксперимент2', 'Теоретическое значение', 'Отклонение1', 'Отклонение2'])
table.add_row(["M", M_1, M_2, M_theor, abs(M_1 - M_theor), abs(M_2 - M_theor)])
table.add_row(["D", D_1, D_2, D_theor, abs(D_1 - D_theor), abs(D_2 - D_theor)])
print(table)

get_plots(array_1)
get_plots(array_2)
