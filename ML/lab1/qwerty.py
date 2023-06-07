import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('glass.csv')
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=123)

acc = []
for i in range(1, 20):
    clf = BaggingClassifier(estimator=KNeighborsClassifier(n_neighbors=1, metric='braycurtis'),
                            n_estimators=i, ).fit(X_train, Y_train)
    acc.append(clf.score(X_test, Y_test))

plt.plot(range(1, 20), acc)
plt.show()

clasif = KNeighborsClassifier(n_neighbors=1, metric='braycurtis')
clasif.fit(X_train, Y_train)
print(clf.score(X_test, Y_test))
