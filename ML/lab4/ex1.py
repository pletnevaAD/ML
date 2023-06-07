import matplotlib.pyplot as plt
import pandas as pd
from sklearn import decomposition
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('glass.csv')
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

classifiers = [SVC(), GaussianNB(), KNeighborsClassifier(n_neighbors=4),
               DecisionTreeClassifier()]

accuracy = {}
for classifier in classifiers:
    accuracy_array = []
    for n_estimators in range(1, 100):
        clf = BaggingClassifier(estimator=classifier,
                                n_estimators=n_estimators, random_state=0).fit(X_train, Y_train)
        accuracy_array.append(clf.score(X_test, Y_test))
    accuracy[classifier] = accuracy_array

fig, ax = plt.subplots(4, 1)
fig.set_figheight(10)
fig.set_figwidth(7)
plt.subplots_adjust(wspace=0.55, hspace=0.55)
for clf, array in accuracy.items():
    ax[list(accuracy).index(clf)].plot(range(1, 100), array)
    ax[list(accuracy).index(clf)].set_title(clf)

plt.show()
