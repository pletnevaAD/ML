import pandas as pd
from matplotlib import pyplot as plt
from sklearn import cluster
from sklearn.cluster import DBSCAN, AgglomerativeClustering


def visualized_fun(data, cluster_num):
    kmeans = cluster.KMeans(cluster_num)
    dbscan = DBSCAN()
    kmeans.fit(data)
    dbscan.fit(data)
    agg_clustering = AgglomerativeClustering(cluster_num)
    agg_clustering.fit(data)
    fig, ax = plt.subplots(3, 1)
    ax[0].scatter(data[0], data[1], c=kmeans.labels_)
    ax[0].set_title('K-Means')
    ax[1].scatter(data[0], data[1], c=dbscan.labels_)
    ax[1].set_title('DBSCAN')
    ax[2].scatter(data[0], data[1], c=agg_clustering.labels_)
    ax[2].set_title('Agglomerative Clustering')
    plt.show()


data_1 = pd.read_csv("clustering_1.csv", header=None, sep="\t")
data_2 = pd.read_csv("clustering_2.csv", header=None, sep="\t")
data_3 = pd.read_csv("clustering_3.csv", header=None, sep="\t")

visualized_fun(data_1, 2)
visualized_fun(data_2, 3)
visualized_fun(data_3, 2)
