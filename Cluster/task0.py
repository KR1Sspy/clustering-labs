
import numpy as np
import clustering

## Generate the dataset to be clustered and print it (it should 100 rows and 2 columns)
X = 10 * np.random.randn(100, 2) + 6
print("data: ")
print(X)

print("k-means clustering")
clustering.plot_data(X)

# run the k-means algorithm with N clusters
X, kmeans, silhouette_score, euclid_dist = clustering.clustering_kmeans(X, 4)
# plot the results
clustering.plot_data_with_clusters(X, kmeans, True)

## TODO: Compare the results with different number of clusters. What do you observe?