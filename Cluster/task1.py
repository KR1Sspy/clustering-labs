
import numpy as np
import matplotlib.pyplot as plt
import clustering
import file_manager

# read the dataset from the CSV file
X, head = file_manager.read_csv_file('data/countries_continents.csv')

print("data: ")
print(X)
print("k-means clustering")

# get the unique continents from the dataset
print("continents: ")
continents = {e: i for i, e in enumerate(np.unique(X[:, 3]))}
print(continents)

# map text data to numbers
X[:, 3] = np.vectorize(continents.__getitem__)(X[:, 3])
X = X[:, 1: np.size(X, 1)-1]

# run the k-means algorithm with nc (number of continents) clusters
X, kmeans, silhouette_score, euclid_dist = clustering.clustering_kmeans(X, 4)

# plot the results
clustering.plot_data_with_clusters(X, kmeans, True, head[1], head[2], False)
# add 4 squares delimiter
plt.axvline(x=np.min(X[:, 0] + (np.max(X[:, 0]) - np.min(X[:, 0])) / 2))
plt.axhline(y=0)
plt.show()

## TODO: Compare the results with different number of clusters. What do you observe?

