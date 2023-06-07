import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm, metrics
from sklearn.inspection import DecisionBoundaryDisplay


def get_plot(X, classif, Y):
    fig, ax = plt.subplots()
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X.iloc[:, 0], X.iloc[:, 1]

    disp = DecisionBoundaryDisplay.from_estimator(
        classif,
        X,
        response_method="predict",
        cmap=plt.cm.coolwarm,
        alpha=0.8,
        ax=ax
    )
    ax.scatter(X0, X1, c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    ax.set_xticks(())
    ax.set_yticks(())


def read_txt(file_name):
    df = pd.read_csv(file_name, sep='\t')
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    return X, Y


X_train, Y_train = read_txt('svmdata_a.txt')
X_test, Y_test = read_txt('svmdata_a_test.txt')
clf = svm.SVC(kernel='linear')
clf.fit(X_train, Y_train)
print(clf.n_support_)
Y_pred_test = clf.predict(X_test)
matrix_test = metrics.confusion_matrix(Y_test, Y_pred_test)
Y_pred_train = clf.predict(X_train)
matrix_train = metrics.confusion_matrix(Y_train, Y_pred_train)
print(f"Матрица ошибок для обучающей выборки:\n{matrix_train}\nМатрица ошибок для тестовой выборки:\n{matrix_test}")
get_plot(X_train, clf, Y_train)
plt.show()

X_train, Y_train = read_txt('svmdata_b.txt')
X_test, Y_test = read_txt('svmdata_b_test.txt')
train_score = []
test_score = []
flag_test = False
flag_train = False
for param_c in range(1, 500):
    clf = svm.SVC(kernel='linear', C=param_c)
    clf.fit(X_train, Y_train)
    train_score.append(clf.score(X_train, Y_train))
    test_score.append(clf.score(X_test, Y_test))
    if train_score[-1] == 1 and flag_train is False:
        print("Train C = ", param_c)
        flag_train = True
    if test_score[-1] < 1 and flag_test is False:
        print("Test C = ", param_c)
        flag_test = True

plt.plot(range(1, 500), train_score)
plt.plot(range(1, 500), test_score)
plt.legend(('train', 'test'))
plt.show()


def different_kernels_visualize(X_train, Y_train):
    kernels = ['linear', 'sigmoid']

    for ker in kernels:
        svc = svm.SVC(kernel=ker)
        svc.fit(X_train, Y_train)
        get_plot(X_train, svc, Y_train)
        plt.title(f"Kernel: {ker}")
        plt.show()

    svc = svm.SVC(kernel="rbf", gamma=0.1)
    svc.fit(X_train, Y_train)
    get_plot(X_train, svc, Y_train)
    plt.title(f"Kernel: Gauss")
    plt.show()

    degrees = [1, 2, 3, 4, 5]
    for deg in degrees:
        svc = svm.SVC(kernel='poly', degree=deg)
        svc.fit(X_train, Y_train)
        get_plot(X_train, svc, Y_train)
        plt.title(f"Kernel: poly, degree: {deg}")
        plt.show()


X_train, Y_train = read_txt('svmdata_c.txt')
X_test, Y_test = read_txt('svmdata_c_test.txt')

# different_kernels_visualize(X_train, Y_train)

X_train, Y_train = read_txt('svmdata_d.txt')
X_test, Y_test = read_txt('svmdata_d_test.txt')
# different_kernels_visualize(X_train, Y_train)

X_train, Y_train = read_txt('svmdata_e.txt')
X_test, Y_test = read_txt('svmdata_e_test.txt')

gamma = ['scale', 500]

for gam in gamma:
    kernels = ['sigmoid']

    for ker in kernels:
        svc = svm.SVC(kernel=ker)
        svc.fit(X_train, Y_train)
        get_plot(X_train, svc, Y_train)
        plt.title(f"Kernel: {ker}, gamma: {gam}")
        plt.show()

    svc = svm.SVC(kernel="rbf", gamma=gam)
    svc.fit(X_train, Y_train)
    get_plot(X_train, svc, Y_train)
    plt.title(f"Kernel: Gauss, gamma: {gam}")
    plt.show()

    degrees = [1, 2, 3, 4, 5]
    for deg in degrees:
        svc = svm.SVC(kernel='poly', degree=deg, gamma=gam)
        svc.fit(X_train, Y_train)
        get_plot(X_train, svc, Y_train)
        plt.title(f"Kernel: poly, degree: {deg}, gamma:{gam}")
        plt.show()
