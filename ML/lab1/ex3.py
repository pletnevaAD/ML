import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("glass.csv")

X = pd.get_dummies(df.iloc[:, 1:-1])
Y = df.iloc[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.2, random_state=123)

all_metrics = ["braycurtis",
               "canberra",
               "chebyshev",
               "cityblock",
               "correlation",
               "cosine",
               "euclidean",
               "minkowski",
               "sqeuclidean"]
max_accuracy_metrics = {}
for metric in all_metrics:
    accuracy = []
    for n_neigh in range(1, 100):
        neigh = KNeighborsClassifier(n_neighbors=n_neigh, metric=metric)
        neigh.fit(X_train, Y_train)
        accuracy.append(neigh.score(X_test, Y_test))
    max_accuracy_metrics[metric] = max(accuracy)

max_key = max(max_accuracy_metrics, key=lambda k: max_accuracy_metrics[k])
print('Метрики и максимальные точности:', max_accuracy_metrics)
print('Оптимальная метрика:', max_key)
accuracy = []
for n_neigh in range(1, 100):
    neigh = KNeighborsClassifier(n_neighbors=n_neigh, metric=max_key)
    neigh.fit(X_train, Y_train)
    accuracy.append(neigh.score(X_test, Y_test))

print(accuracy)

plt.plot(range(1, 100), accuracy)
plt.show()

neigh = KNeighborsClassifier(n_neighbors=2, metric='correlation')
neigh.fit(X_train, Y_train)
Y_predict = neigh.predict([[1.516, 11.7, 1.01, 1.19, 72.59, 0.43, 11.44, 0.02, 0.1]])
print(Y_predict)
