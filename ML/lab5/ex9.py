import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from pandas import DataFrame
from sklearn.linear_model import Ridge, LinearRegression

data = pd.read_csv('nsw74psid1.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

svr = SVR().fit(X_train, y_train)
print(f'R^2 SVR: {svr.score(X_test, y_test)}')

lr = LinearRegression().fit(X_train, y_train)
print(f'R^2 LinearRegression: {lr.score(X_test, y_test)}')

dtr = DecisionTreeRegressor().fit(X_train, y_train)
print(f'R^2 DecisionTreeRegressor: {dtr.score(X_test, y_test)}')
