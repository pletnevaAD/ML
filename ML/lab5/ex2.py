from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

data = pd.read_csv('reglab.txt', sep='\t')
Y = data.iloc[:, 0]
all_x = [data.iloc[:, i] for i in range(1, 5)]

for i in range(4):
    print(f'k={i+1}')
    for comb in combinations(all_x, i+1):
        features = []
        for feature in comb:
            features.append(feature)
        df_x = pd.concat(features, axis=1)
        reg = linear_model.LinearRegression().fit(df_x, Y)
        RSS = mean_squared_error(Y, reg.predict(df_x)) * len(Y)
        print(f'{[df_x.columns[j] for j in range(i+1)]}: RSS={RSS}')


