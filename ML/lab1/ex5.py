import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz


def accuracy_plot(X_train, Y_train, X_test, Y_test):
    criteria = ['gini', 'entropy']
    for crit in criteria:
        accuracies = []
        for i in range(1, 51):
            tree_clf = DecisionTreeClassifier(random_state=0, max_depth=i, criterion=crit)
            tree_clf.fit(X_train, Y_train)
            accuracies.append(tree_clf.score(X_test, Y_test))
        plt.plot(range(1, 51), accuracies)

    plt.title("Зависимость точности от максимальной глубины")
    plt.legend(('gini', 'entropy'))
    plt.show()


df = pd.read_csv("glass.csv")

X = pd.get_dummies(df.iloc[:, 1:-1])
Y = df.iloc[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.3, random_state=123)

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, Y_train)
print("Accuracy without max_depth arg:", clf.score(X_test, Y_test), "\nTree depth:", clf.get_depth())

fig = plt.figure(figsize=(20, 12))
_ = tree.plot_tree(clf,
                   filled=True, fontsize=11)
plt.show()

clf2 = DecisionTreeClassifier(random_state=0, max_depth=4)
clf2.fit(X_train, Y_train)
print("Accuracy with max_depth=4:", clf2.score(X_test, Y_test), "\nTree depth:", clf2.get_depth())

accuracy_plot(X_train, Y_train, X_test, Y_test)

df = pd.read_csv("spam7.csv")

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.3, random_state=123)

accuracy_plot(X_train, Y_train, X_test, Y_test)
tree_clf = DecisionTreeClassifier(random_state=0, max_depth=8)
tree_clf.fit(X_train, Y_train)
print(tree_clf.score(X_test, Y_test))
_ = tree.plot_tree(tree_clf,
                   filled=True, fontsize=4)
plt.show()

