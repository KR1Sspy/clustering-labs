
import numpy as np
import matplotlib.pyplot as plt
import clustering

## Generate the dataset to be clustered and print it (it should 100 rows and 2 columns)
X = 1 * np.random.randn(100, 2) + 6
print("data: ")
print(X)

print("k-means clustering")
# clustering.plot_data(X)

r = range(2, 20)
silhouette_score_vect = []
WCSS_vect = []

optimal_number_of_clusters = 0
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
plt.plot(r, silhouette_score_vect)
plt.title("Clustering evaluation")
plt.legend(["silhouette score"])
plt.xticks(r)
plt.subplot(212)
plt.plot(r, WCSS_vect)
plt.legend(["WCSS"])
plt.xticks(r)
plt.show()
