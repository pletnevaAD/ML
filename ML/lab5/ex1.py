import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split

data = pd.read_csv('reglab1.txt', sep='\t')

Z = data.iloc[:, 0]
X = data.iloc[:, 1]
Y = data.iloc[:, 2]
ZX = data.iloc[:, :-1]
ZY = data.iloc[:, [0, 2]]
XY = data.iloc[:, 1:]
array_xyz = [Z, X, Y]

for i in range(len(array_xyz)):
    for j in range(len(array_xyz)):
        if i != j:
            X_train, X_test, y_train, y_test = train_test_split(array_xyz[i], array_xyz[j], test_size=0.2,
                                                                random_state=1)
            reg = linear_model.LinearRegression()

            reg.fit(np.array(X_train).reshape(-1, 1), np.array(y_train))
            print(
                f'Coef R^2 {y_train._name}({X_train._name}): {reg.score(np.array(X_test).reshape(-1, 1), np.array(y_test))}')


X_train, X_test, y_train, y_test = train_test_split(ZY, X, test_size=0.2,
                                                    random_state=1)
reg = linear_model.LinearRegression()
reg.fit(np.array(X_train), np.array(y_train))
print(
    f'Coef R^2 {y_train._name}({X_train.columns[0]},{X_train.columns[1]}): {reg.score(np.array(X_test), np.array(y_test))}')

X_train, X_test, y_train, y_test = train_test_split(ZX, Y, test_size=0.2,
                                                    random_state=1)
reg.fit(np.array(X_train), np.array(y_train))
print(
    f'Coef R^2 {y_train._name}({X_train.columns[0]},{X_train.columns[1]}): {reg.score(np.array(X_test), np.array(y_test))}')

X_train, X_test, y_train, y_test = train_test_split(XY, Z, test_size=0.2,
                                                    random_state=1)
reg.fit(np.array(X_train), np.array(y_train))
print(
    f'Coef R^2 {y_train._name}({X_train.columns[0]},{X_train.columns[1]}): {reg.score(np.array(X_test), np.array(y_test))}')

plt.scatter(X, Y)
plt.title("y(x)")
plt.show()
plt.scatter(Y, X)
plt.title("x(y)")
plt.show()

