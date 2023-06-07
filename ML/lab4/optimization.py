import math

import numpy as np

N = 3287
m = 3
lambdas = [40 * (10 ** (-6)), 10 * (10 ** (-6)), 80 * (10 ** (-6))]
n_i = [3, 2, 6]
P = 0.999
T = 8760


def LFRS(x):
    return (((x[0] > T) and (x[1] > T)) or (x[2] > T)) and (x[3] > T) and (x[4] > T) and (
            ((x[5] > T) and (x[6] > T)) or ((x[7] > T) and (x[8] > T)) or ((x[9] > T) and (x[10] > T)))


def vbr(L):
    d = 0
    for k in range(N):
        x = []
        for i in range(m):
            t = []
            for j in range(n_i[i]):
                t.append((-1) * math.log(np.random.random()) / lambdas[i])
            for j in range(L[i]):
                index = t.index(min(t))
                t[index] += (-1) * math.log(np.random.random()) / lambdas[i]
            x.extend(t)
        if not LFRS(x):
            d += 1
    return 1 - d / N


for i in range(6):
    for j in range(6):
        for k in range(10):
            p = vbr([i, j, k])
            if p > P:
                print(f'Количество запасных частей для каждого типа элементов: {[i, j, k]}\n'
                      f'Вероятность безотказной работы: {p}')
