import pandas as pd
from matplotlib import pyplot as plt
from numpy import array
from scipy.cluster import hierarchy
from sklearn import decomposition

data = pd.read_csv('votes.csv')
data = data.dropna(how='any')
temp = hierarchy.linkage(array(data))
plt.figure()
dn = hierarchy.dendrogram(temp)
plt.show()

pca = decomposition.PCA(n_components=2)
X_reduced = pca.fit_transform(data)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.show()

