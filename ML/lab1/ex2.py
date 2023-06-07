from collections import namedtuple

import numpy as np
from matplotlib import pyplot as plt
from numpy import random
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

Points = namedtuple('Points', ['x', 'y'])
class_one = Points(random.normal(8, 3, 60), random.normal(12, 3, 60))
class_minus_one = Points(random.normal(21, 4, 40), random.normal(25, 4, 40))

plt.scatter(class_one.x, class_one.y, c='green')
plt.scatter(class_minus_one.x, class_minus_one.y, c='red')
plt.show()

X, Y = np.array(list(zip(class_one.x, class_one.y)) +
                list(zip(class_minus_one.x, class_minus_one.y))), \
    [1] * 60 + [-1] * 40
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.2, random_state=123)
gaussian_nb = GaussianNB()
gaussian_nb.fit(X_train, Y_train)
accuracy = gaussian_nb.score(X_test, Y_test)
print("Точность:", accuracy)
Y_pred = gaussian_nb.predict(X_test)
matrix = metrics.confusion_matrix(Y_test, Y_pred)
print("Матрица ошибок:\n", matrix)
fpr, tpr, _ = metrics.roc_curve(Y_test, Y_pred)

# create ROC curve
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1])
plt.legend(('Practical classifier', 'Random Guess'))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

precision, recall, thresholds = precision_recall_curve(Y_test, Y_pred)

#create precision recall curve
plt.plot(recall, precision, color='purple')

#add axis labels to plot
plt.ylabel('Precision')
plt.xlabel('Recall')

#display plot
plt.show()
