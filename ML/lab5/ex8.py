import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from pandas import DataFrame
from sklearn.linear_model import Ridge, LinearRegression

data = pd.read_csv('svmdata6.txt', sep='\t')
X = np.array(data['X']).reshape(-1, 1)
y = data['Y']
eps = [i * 0.01 for i in range(0, 101)]
mse = []
for epsilon in eps:
    reg = SVR(C=1, epsilon=epsilon).fit(X, y)
    mse.append(mean_squared_error(y, reg.predict(X)))

plt.plot(eps, mse)
plt.show()
