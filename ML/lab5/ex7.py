import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import Ridge, LinearRegression

data = pd.read_csv('cars.csv', sep=',')
speed = np.array(data['speed']).reshape(-1, 1)
dist = data['dist']
reg = LinearRegression().fit(speed, dist)
print(f'Длина тормозного пути при скорости 40 миль в час: {reg.predict(np.array([40]).reshape(-1, 1))}')
plt.scatter(speed, dist, color='black')
plt.plot(speed, reg.predict(speed), color='red', linewidth=4)
plt.show()
