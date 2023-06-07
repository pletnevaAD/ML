import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

df_0 = pd.read_csv("nn_0.csv")
df_1 = pd.read_csv("nn_1.csv")

X_0 = df_0.iloc[:, :-1]
Y_0 = df_0.iloc[:, -1]

X_1 = df_1.iloc[:, :-1]
Y_1 = df_1.iloc[:, -1]

fig, ax = plt.subplots(1, 2)
ax[0].scatter(X_0.X1, X_0.X2, c=Y_0)
ax[0].set_title("nn_0")
ax[1].scatter(X_1.X1, X_1.X2, c=Y_1)
ax[1].set_title("nn_1")
plt.show()

X_0_train, X_0_test, Y_0_train, Y_0_test = train_test_split(X_0, Y_0, test_size=0.2, random_state=42)
X_1_train, X_1_test, Y_1_train, Y_1_test = train_test_split(X_1, Y_1, test_size=0.2, random_state=42)

optimizer = ['lbfgs', 'sgd', 'adam']
activation = ['identity', 'logistic', 'tanh', 'relu']

accuracy_clf_0 = {}
accuracy_clf_1 = {}
for func in activation:
    value_clf_0 = []
    value_clf_1 = []
    for optim in optimizer:
        clf_0 = MLPClassifier(random_state=42, hidden_layer_sizes=(1,), activation=func, solver=optim, max_iter=200).fit(X_0_train,
                                                                                                          Y_0_train)
        value_clf_0.append((optim, clf_0.score(X_0_test, Y_0_test)))
        clf_1 = MLPClassifier(random_state=42, hidden_layer_sizes=(1,), activation=func, solver=optim, max_iter=200).fit(X_1_train,
                                                                                                          Y_1_train)
        value_clf_1.append((optim, clf_1.score(X_1_test, Y_1_test)))
    accuracy_clf_0[func] = value_clf_0
    accuracy_clf_1[func] = value_clf_1

print(accuracy_clf_0)
print(accuracy_clf_1)

fig, ax = plt.subplots(4, 2)
fig.set_figheight(10)
fig.set_figwidth(7)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
for key, value in accuracy_clf_0.items():
    data = []
    for element in value:
        data.append(element[1])
    ax[activation.index(key), 0].bar(optimizer, data)
    ax[activation.index(key), 0].set_ylim(0, 1)
    ax[activation.index(key), 0].set_title(f'nn_0 with activation func {key}')
for key, value in accuracy_clf_1.items():
    data = []
    for element in value:
        data.append(element[1])
    ax[activation.index(key), 1].bar(optimizer, data)
    ax[activation.index(key), 1].set_ylim(0, 1)
    ax[activation.index(key), 1].set_title(f'nn_1 with activation func {key}')

plt.show()

accuracy_0 = []
accuracy_1 = []
for i in range(1, 100):
    clf_0 = MLPClassifier(random_state=42, hidden_layer_sizes=(1,), max_iter=i, solver='lbfgs',
                          activation='tanh').fit(X_0_train, Y_0_train)
    accuracy_0.append(clf_0.score(X_0_test, Y_0_test))
    clf_1 = MLPClassifier(random_state=42, hidden_layer_sizes=(1,), max_iter=i, solver='lbfgs',
                          activation='tanh').fit(X_1_train, Y_1_train)
    accuracy_1.append(clf_1.score(X_1_test, Y_1_test))

print(accuracy_0)
print(accuracy_1)

fig, ax = plt.subplots(2, 1)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
ax[0].plot(range(1, 100), accuracy_0)
ax[0].set_title('Зависимость точности от количества эпох для nn_0')
ax[1].plot(range(1, 100), accuracy_1)
ax[1].set_title('Зависимость точности от количества эпох для nn_1')
plt.show()
