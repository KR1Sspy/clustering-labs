import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

def plot_data(X):
    # Let's see the points generated more visual. Plot the points using matplotlib
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()


def plot_data_with_clusters(X, kmeans, show_centers=False, xlabel=None, ylabel=None, show=True):
    # Let's see the clusters more visual. Let's plot the graphic for our points and draw each point to the cluster it is assigned to
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_,
                cmap='rainbow', label="points")

    if show_centers:
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
                    :, 1], color="orange", s=200, label="centroids", marker="o")

    plt.title('K-Means Clustering')
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.legend(loc='upper left')
    plt.xticks()
    if show:
        plt.show()


def clustering_kmeans(X, k):
    kmeans = KMeans(n_clusters=k, init="random")
    X_input = np.reshape(X, (-1, 1))
    X_input = X
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(X_input)
    clusters = kmeans.predict(X_input)

    print("cluster labels: ")
    print(clusters)

    # Let's print the cluster centers
    print("cluster centers")
    centroids = kmeans.cluster_centers_
    print(centroids)

    # Let's print the labels for our points. In this context, label is the cluster on which the data point is assigned to after the clusterring process
    print("cluster labels")
    print(kmeans.labels_)

    sscore = silhouette_score(X_input, clusters)
    print("For n_clusters =", k,
          "The average silhouette_score is :", sscore)

    WCSS = kmeans.inertia_
    return X, kmeans, sscore, WCSS
