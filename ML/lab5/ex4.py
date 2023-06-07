import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

data = pd.read_csv('longley.csv', sep=',')
data = data.drop(columns=['Population'])
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=10)
reg = linear_model.LinearRegression().fit(X_train, y_train)
print(f'MSE на тестовой выборке для линейной регрессии: {mean_squared_error(y_test, reg.predict(X_test))}')
print(f'MSE на обучающей выборке для линейной регрессии: {mean_squared_error(y_train, reg.predict(X_train))}')

r2_test = []
r2_train = []
lambdas = [10 ** (-3 + 0.2 * i) for i in range(26)]
for lambda_ in lambdas:
    ridge_reg = Ridge(alpha=lambda_).fit(X_train, y_train)
    r2_test.append(mean_squared_error(y_test, ridge_reg.predict(X_test)))
    r2_train.append(mean_squared_error(y_train, ridge_reg.predict(X_train)))

plt.plot(lambdas, r2_test, color='red')
plt.plot(lambdas, r2_train, color='blue')
plt.legend(['test', 'train'])
plt.show()
