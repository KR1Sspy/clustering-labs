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

# define the range of clusters
r = range(2, 10)
silhouette_score_vect = []
WCSS_vect = []

optimal_number_of_clusters = 2
max_silhouette_score = 0

for i in r:
    # run the k-means algorithm with i clusters
    X, kmeans, silhouette_score, WCSS = clustering.clustering_kmeans(X, i)
    # save the results into a list
    silhouette_score_vect.append(silhouette_score)
    WCSS_vect.append(WCSS)
    # check the optimal number of clusters by silhouette score
    if silhouette_score > max_silhouette_score:
        max_silhouette_score = silhouette_score
        optimal_number_of_clusters = i

# plot the results
print("optimal number of clusters: ", optimal_number_of_clusters)
plt.subplot(211)
plt.plot(r, silhouette_score_vect, 'bo-')
plt.title("Clustering evaluation")
plt.legend(["Silhouette Score"])
plt.xticks(r)
plt.grid(True, alpha=0.3)

plt.subplot(212)
plt.plot(r, WCSS_vect, 'ro-')
plt.legend(["WCSS"])
plt.xticks(r)
plt.xlabel("k (number of clusters)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()