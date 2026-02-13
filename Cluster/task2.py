
import numpy as np
import matplotlib.pyplot as plt
import clustering
import file_manager
from sklearn import preprocessing

# read the dataset from the CSV file
X, head = file_manager.read_csv_file('data/market_segmentation_data.csv')

print("data: ")
print(X)
print("k-means clustering")

# scale the data
X = preprocessing.scale(X)

# run the k-means algorithm with 2 clusters
X, kmeans, silhouette_score, euclid_dist = clustering.clustering_kmeans(X, 4)

# plot the results
clustering.plot_data_with_clusters(X, kmeans, True, head[0], head[1], False)
# add 4 squares delimiter
plt.axvline(x=np.min(X[:, 0] + (np.max(X[:, 0]) - np.min(X[:, 0]))/2))
plt.axhline(y=0)
plt.show()