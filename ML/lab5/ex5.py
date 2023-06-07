import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

data = pd.read_csv('eustock.csv', sep=',')
DAX = data['DAX']
SMI = data['SMI']
CAC = data['CAC']
FTSE = data['FTSE']

x = [i for i in range(len(data))]
plt.plot(x, DAX)
plt.plot(x, SMI)
plt.plot(x, CAC)
plt.plot(x, FTSE)

plt.legend(['DAX', 'SMI', 'CAC', 'FTSE'])

plt.show()
x = np.array(x).reshape(-1, 1)
reg_DAX = LinearRegression().fit(x, DAX)
reg_SMI = LinearRegression().fit(x, SMI)
reg_CAC = LinearRegression().fit(x, CAC)
reg_FTSE = LinearRegression().fit(x, FTSE)
reg_all = LinearRegression().fit(x, data)

fig, ax = plt.subplots(2, 2)
fig.subplots_adjust(hspace=0.4)

ax[0][0].plot(x, DAX)
ax[0][0].plot(x, reg_DAX.predict(x), linewidth=3)
ax[0][0].set_title('DAX')

ax[0][1].plot(x, SMI)
ax[0][1].plot(x, reg_SMI.predict(x), linewidth=3)
ax[0][1].set_title('SMI')

ax[1][0].plot(x, CAC)
ax[1][0].plot(x, reg_CAC.predict(x), linewidth=3)
ax[1][0].set_title('CAC')

ax[1][1].plot(x, FTSE)
ax[1][1].plot(x, reg_FTSE.predict(x), linewidth=3)
ax[1][1].set_title('FTSE')

plt.show()

