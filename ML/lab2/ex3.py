from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils import check_random_state

X, y = fetch_openml(
    "mnist_784", version=1, return_X_y=True, as_frame=False, parser="pandas"
)

random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
)

accuracy = []
for i in range(1, 11):
    clf_1 = MLPClassifier(random_state=42, hidden_layer_sizes=(64,), max_iter=i).fit(X_train, y_train)
    accuracy.append(clf_1.score(X_test, y_test))

for acc in accuracy:
    print(f'Max_iter={accuracy.index(acc) + 1}:', acc)
