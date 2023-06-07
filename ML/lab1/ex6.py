import pandas as pd
from sklearn import metrics, svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from lab1.ex5 import accuracy_plot

df_train = pd.read_csv("bank_scoring_train.csv", sep='\t')

X_train = df_train.iloc[:, 1:]
Y_train = df_train.iloc[:, 0]

df_test = pd.read_csv("bank_scoring_test.csv", sep='\t')

X_test = df_test.iloc[:, 1:]
Y_test = df_test.iloc[:, 0]

gaussian_nb = GaussianNB()
gaussian_nb.fit(X_train, Y_train)

Y_pred = gaussian_nb.predict(X_test)

print("Матрица ошибок для наивного Байесовского классификатора:")
tn, fp, fn, tp = metrics.confusion_matrix(Y_test, Y_pred).ravel()
print(tp, fp)
print(fn, tn)

# accuracy = {}
# for n_neigh in range(1, 100):
#     neigh = KNeighborsClassifier(n_neighbors=n_neigh)
#     neigh.fit(X_train, Y_train)
#     accuracy[n_neigh] = neigh.score(X_test, Y_test)
#
# max_key = max(accuracy, key=lambda k: accuracy[k])
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X_train, Y_train)
Y_pred = neigh.predict(X_test)

print(f"Матрица ошибок для метода k ближайших соседей при k={1}:")
tn, fp, fn, tp = metrics.confusion_matrix(Y_test, Y_pred).ravel()
print(tp, fp)
print(fn, tn)

# kernels = ['linear', 'sigmoid']
#
# for ker in kernels:
#     svc = svm.SVC(kernel=ker)
#     svc.fit(X_train, Y_train)
#     Y_pred = svc.predict(X_test)
#     print(f"Матрица ошибок для метода SVM при kernel={ker}:")
#     tn, fp, fn, tp = metrics.confusion_matrix(Y_test, Y_pred).ravel()
#     print(tp, fp)
#     print(fn, tn)

# svc = svm.SVC(kernel="rbf", gamma=0.1)
# svc.fit(X_train, Y_train)
# Y_pred = svc.predict(X_test)
# print(f"Матрица ошибок для метода SVM при kernel=rbf:")
# tn, fp, fn, tp = metrics.confusion_matrix(Y_test, Y_pred).ravel()
# print(tp, fp)
# print(fn, tn)
#
# degrees = [2, 3]
# for deg in degrees:
#     svc = svm.SVC(kernel='poly', degree=deg)
#     svc.fit(X_train, Y_train)
#     Y_pred = svc.predict(X_test)
#     print(f"Матрица ошибок для метода SVM при kernel=poly степени {deg}:")
#     tn, fp, fn, tp = metrics.confusion_matrix(Y_test, Y_pred).ravel()
#     print(tp, fp)
#     print(fn, tn)

# accuracy_plot(X_train, Y_train, X_test, Y_test)
tree_clf = DecisionTreeClassifier(random_state=0, max_depth=34)
tree_clf.fit(X_train, Y_train)
Y_pred = tree_clf.predict(X_test)

print(f"Матрица ошибок для метода решающих деревьев при максимальной глубине 40 и критерии расщепления gini:")
tn, fp, fn, tp = metrics.confusion_matrix(Y_test, Y_pred).ravel()
print(tp, fp)
print(fn, tn)
