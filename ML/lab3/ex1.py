import matplotlib.pyplot as plt
import pandas as pd
from sklearn import cluster, decomposition
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler


def clustering(data, max_iteration, title):
    scores = {}
    for num_iter in range(1, max_iteration):
        kmeans = cluster.KMeans(3, max_iter=num_iter)
        kmeans.fit(data)
        scores[num_iter] = davies_bouldin_score(data, kmeans.labels_)
    max_key = max(scores, key=lambda k: scores[k])
    kmeans = cluster.KMeans(3, max_iter=max_key)
    kmeans.fit(data)
    pca = decomposition.PCA(n_components=2)
    X_reduced = pca.fit_transform(data)
    fig, ax = plt.subplots(2, 1)
    fig.suptitle(title)
    ax[0].scatter(X_reduced[:, 0], X_reduced[:, 1], c=kmeans.labels_)
    ax[1].plot(scores.keys(), scores.values())
    plt.show()


data = pd.read_csv('pluton.csv')

scaler = StandardScaler()
scale_data = scaler.fit_transform(data)

clustering(data, 50, "Кластеризация данных")

clustering(scale_data, 50, "Кластеризация стандартизированных данных")