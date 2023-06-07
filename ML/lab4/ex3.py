import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# загрузка данных
traindf = pd.read_csv('titanic_train.csv')

# удаление ненужных столбцов
traindf.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
imputer = SimpleImputer(strategy='median')
traindf['Age'] = imputer.fit_transform(traindf[['Age']])
X = pd.get_dummies(traindf.drop('Survived', axis=1))
y = pd.factorize(traindf['Survived'])[0]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# определение моделей для использования в стекинге
estimators = [('gnb', GaussianNB()), ('svc', SVC()), ('dt', DecisionTreeClassifier()), ('knn', KNeighborsClassifier())]

# создание классификатора со стекингом моделей

clfs = [StackingClassifier(estimators=[('gnb', GaussianNB())]), StackingClassifier(estimators=[('svc', SVC())]),
        StackingClassifier(estimators=[('dt', DecisionTreeClassifier())]),
        StackingClassifier(estimators=[('knn', KNeighborsClassifier())]),
        StackingClassifier(estimators=estimators)]

for clf in clfs:
    clf.fit(X_train, Y_train)

    print(f"Для классификатора {clf} точность: {clf.score(X_test, Y_test)}")
