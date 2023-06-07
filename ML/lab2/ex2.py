import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier


df_1 = pd.read_csv("nn_1.csv")

X_1 = df_1.iloc[:, :-1]
Y_1 = df_1.iloc[:, -1]

X_1_train, X_1_test, Y_1_train, Y_1_test = train_test_split(X_1, Y_1, test_size=0.2, random_state=1)

clf_1 = MLPClassifier(random_state=42, max_iter=50, hidden_layer_sizes=(14, 13)).fit(X_1_train, Y_1_train)
# params = {
#     'hidden_layer_sizes': [(i, j) for i in range(1, 15) for j in range(1, 15)],
# }
# grid_search = GridSearchCV(clf_1, params).fit(X_1_test, Y_1_test)
# print(grid_search.best_params_)
print(clf_1.score(X_1_test, Y_1_test))
optimizer = ['lbfgs', 'sgd', 'adam']
activation = ['identity', 'logistic', 'tanh', 'relu']

accuracy_clf_1 = {}
for func in activation:
    value_clf_1 = []
    for optim in optimizer:
        clf_1 = MLPClassifier(random_state=42, hidden_layer_sizes=(14, 13), activation=func, solver=optim,
                              max_iter=50).fit(X_1_train, Y_1_train)
        value_clf_1.append((optim, clf_1.score(X_1_test, Y_1_test)))
    accuracy_clf_1[func] = value_clf_1

fig, ax = plt.subplots(4, 1)
fig.set_figheight(10)
fig.set_figwidth(5)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
for key, value in accuracy_clf_1.items():
    data = []
    for element in value:
        data.append(element[1])
    ax[activation.index(key)].bar(optimizer, data)
    ax[activation.index(key)].set_ylim(0, 1)
    ax[activation.index(key)].set_title(f'nn_0 with activation func {key}')

plt.show()