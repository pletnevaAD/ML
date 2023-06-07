import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import Ridge, LinearRegression

data = pd.read_csv('JohnsonJohnson.csv', sep=',')
Q1 = DataFrame()
Q2 = DataFrame()
Q3 = DataFrame()
Q4 = DataFrame()
for index, row in data.iterrows():
    if row['index'].endswith('Q1'):
        Q1 = pd.concat([Q1, row], axis=1)
    if row['index'].endswith('Q2'):
        Q2 = pd.concat([Q2, row], axis=1)
    if row['index'].endswith('Q3'):
        Q3 = pd.concat([Q3, row], axis=1)
    if row['index'].endswith('Q4'):
        Q4 = pd.concat([Q4, row], axis=1)

all_Q = [Q1.transpose(), Q2.transpose(), Q3.transpose(), Q4.transpose()]
fig, ax = plt.subplots(2, 1)
fig.subplots_adjust(hspace=0.4)
for q in all_Q:
    x = np.array(q['index'].str[:4].astype(int)).reshape(-1, 1)
    y = np.array(q['value'])
    reg = LinearRegression().fit(x, y)
    print(
        f'Предсказание прибыли для квартала {list(q["index"])[0][5:]} в 2016 году: {reg.predict(np.array([2016]).reshape(-1, 1))}')
    ax[0].plot(x, y)
    ax[1].plot(x, reg.predict(x))

ax[0].set_title('Кривые изменения прибыли во времени для каждого квартала')
ax[1].set_title('Линии регрессии прибыли во времени для каждого квартала')
ax[0].legend(['Q1', 'Q2', 'Q3', 'Q4'])
ax[1].legend(['Q1', 'Q2', 'Q3', 'Q4'])

plt.show()
avg_value = np.mean([q['value'] for q in all_Q], axis=0)
date = [i for i in range(1960, 1981)]
date = np.array(date).reshape(-1, 1)
reg = LinearRegression().fit(date, avg_value)
print(f'Предсказание прибыли в среднем по году в 2016 году: {reg.predict(np.array([2016]).reshape(-1, 1))}')

plt.plot(date, avg_value)
plt.plot(date, reg.predict(date))
plt.title('Средняя прибыль по кварталам')
plt.show()
