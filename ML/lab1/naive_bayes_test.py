from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate


def load_txt(file_name):
    df = pd.read_csv(file_name, header=None)
    X = pd.get_dummies(df.iloc[:, :-1])
    Y = df.iloc[:, -1]
    return X, Y


def load_csv(file_name):
    df = pd.read_csv(file_name)
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    return X, Y


def naive_bayes_result(X_digits, Y_digits, list_percent):
    dict_percent_accuracy = {}
    for percent in list_percent:
        X_train, X_test, Y_train, Y_test = train_test_split(X_digits, Y_digits,
                                                            test_size=percent, random_state=123)
        gaussian_nb = GaussianNB()
        gaussian_nb.fit(X_train, Y_train)
        accuracy = gaussian_nb.score(X_test, Y_test)
        dict_percent_accuracy[percent] = accuracy
    return dict_percent_accuracy


X_tic_tac, Y_tic_tac = load_txt("tic_tac_toe.txt")
print(Y_tic_tac)
test_percent = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
tic_tac_result = naive_bayes_result(X_tic_tac, Y_tic_tac, test_percent)
X_spam, Y_spam = load_csv("spam.csv")
spam_result = naive_bayes_result(X_spam, Y_spam, test_percent)
print(tabulate(tic_tac_result.items(), headers=['Доля тестовых данных', 'Точность']))
print(tabulate(spam_result.items(), headers=['Доля тестовых данных', 'Точность']))

fig, axis = plt.subplots(1, 2)
fig.suptitle("Зависимость точности от доли тестовой части в выборке для:")
axis[0].plot(tic_tac_result.keys(), tic_tac_result.values(), color='red')
axis[0].set_ylim(0, 1)
axis[0].set_title('крестиков-ноликов')
axis[1].plot(spam_result.keys(), spam_result.values(), color='red')
axis[1].set_ylim(0, 1)
axis[1].set_title('спама')

plt.show()
