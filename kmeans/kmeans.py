import numpy as np
import matplotlib.pyplot as plt


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KMeansClustering:
    ''' K-Means Clustering Algorithm implemented with NumPy '''
    def __init__(self, K=3, plot_steps=False, max_iters=300):
        self.K = K
        self.centroids = None
        self.plot_steps = plot_steps
        self.max_iters = max_iters

        self.clusters = [[] for _ in range(self.K)] # List of K lists to hold the data points assigned to each cluster

        self.centroids = [] # List to hold the centroid positions

    def predict(self, X):
        # Predict the closest cluster each data point in X belongs to
        self.X = X
        self.n_samples, self.n_features = X.shape 

        # Initialization
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs] # Randomly select K data points as initial centroids

        # EM algorithm
        # Expectation step
        for _ in range(self.max_iters):
            # Assign samples to closest centroids 
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()
            
            # Maximization step
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # Check for convergence (if centroids do not change)
            if self._is_converged(centroids_old, self.centroids):
                break
            
            if self.plot_steps:
                self.plot()
        
        return self._get_cluster_labels(self.clusters)

    def _create_clusters(self, centroids):
        # Assign the data points to the closest centroids to create clusters
        clusters = [[] for _ in range(self.K)]
        for idx, data_point in enumerate(self.X):
            centroid_idx = self._closest_centroid(data_point, centroids)
            clusters[centroid_idx].append(idx)
        return clusters
    
    def _closest_centroid(self, data_point, centroids):
        # Compute distances between data point and all centroids
        distances = [euclidean_distance(data_point, centroid) for centroid in centroids] # List of distances between one data point and all centroids
        closest_idx = np.argmin(distances) # Index of the closest centroid
        return closest_idx

    def _get_centroids(self, clusters):
        # Recompute centroids as the mean of all data points assigned to each cluster
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
    
        return centroids

    def _is_converged(self, centroids_old, centroids):
        # Check if the centroids have changed
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def _get_cluster_labels(self, clusters):
        # Assign the label of the cluster to each data point
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for data_point in cluster:
                labels[data_point] = cluster_idx
        return labels

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        if self.centroids is not None:
            for point in self.centroids:
                ax.scatter(*point, marker="x", color="black", linewidth=2)
        plt.show()
    # def fit(self, X, max_iters=200):
    #     # Randomly initialize centroids, bounded by data range (uniformly distributed)
    #     self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), 
    #                                        size=(self.k,X.shape[1])) 
    #     y = np.array([])  
    #     for _ in range(max_iters):
    #         # Assign clusters
    #         y = []
    #         # Expectation step
    #         # Compute distances between data points and centroids
    #         for data_point in X:
    #             distances = KMeansClustering.euclidean_distance(data_point, self.centroids) # List of distances between one data point and all centroids
    #             cluster_num = np.argmin(distances) # Index of the closest centroid
    #             y.append(cluster_num) # Assign the data point to the closest centroid's cluster

    #         y = np.array(y) 
    #         # End of cluster assignment to each data point 

    #         # Recompute centroids
    #         cluster_indices = []
    #         for i in range(self.k):
    #             # Which indices belong to cluster i?
    #             cluster_indices.append(np.where(y == i)[0])

    #         # New list to hold the new cluster centres
    #         cluster_centres = []
    #         # Maximization step
    #         # Reposition the centroids
    #         for i, indices in enumerate(cluster_indices):
    #             if len(indices) == 0:
    #                 # If a cluster has no points assigned, keep the centroid unchanged
    #                 cluster_centres.append(self.centroids[i])
    #             else:
    #                 cluster_centres.append(np.mean(X[indices], axis=0)) # New centroid is the mean of all points assigned to the cluster

    #         if np.max(self.centroids - np.array(cluster_centres)) < 0.0001:
    #             # If centroids do not change, we have converged
    #             break
    #         else:
    #             self.centroids = np.array(cluster_centres) # Update centroids for the next iteration
    #     return y