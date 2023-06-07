import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('vehicle.csv')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)
classifiers = [GaussianNB(), SVC(), DecisionTreeClassifier(max_depth=1)]

accuracy_of_clf = {}
for classifier in classifiers:
    if hasattr(classifier, 'predict_proba'):
        accuracy = []
        for i in range(1, 100):
            clf = AdaBoostClassifier(estimator=classifier, n_estimators=i, random_state=0)
            clf.fit(X_train, Y_train)
            accuracy.append(clf.score(X_test, Y_test))
        accuracy_of_clf[classifier] = accuracy

fig, ax = plt.subplots(len(accuracy_of_clf), 1)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
for key, value in accuracy_of_clf.items():
    ax[list(accuracy_of_clf).index(key)].plot(range(1, 100), value)
    ax[list(accuracy_of_clf).index(key)].set_title(key)

plt.show()
