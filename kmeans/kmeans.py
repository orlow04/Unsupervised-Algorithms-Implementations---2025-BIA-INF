import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from math import comb


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))


def rand_index(labels_true, labels_pred):
    """
    Compute the Rand Index between two clusterings.
    
    Parameters:
    -----------
    labels_true : array-like, shape = [n_samples]
        Ground truth class labels
    labels_pred : array-like, shape = [n_samples]
        Cluster labels to evaluate
        
    Returns:
    --------
    float : Rand Index value between 0 and 1
    """
    n = len(labels_true)
    
    # Count pairs
    a = 0  # pairs in same cluster in both labelings
    b = 0  # pairs in different clusters in both labelings
    
    for i in range(n):
        for j in range(i + 1, n):
            same_true = labels_true[i] == labels_true[j]
            same_pred = labels_pred[i] == labels_pred[j]
            
            if same_true and same_pred:
                a += 1
            elif not same_true and not same_pred:
                b += 1
    
    total_pairs = comb(n, 2)
    return (a + b) / total_pairs


def adjusted_rand_index(labels_true, labels_pred):
    """
    Compute the Adjusted Rand Index between two clusterings.
    
    Parameters:
    -----------
    labels_true : array-like, shape = [n_samples]
        Ground truth class labels
    labels_pred : array-like, shape = [n_samples]
        Cluster labels to evaluate
        
    Returns:
    --------
    float : Adjusted Rand Index value, bounded above by 1.0
    """
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    
    # Get unique labels
    true_labels = np.unique(labels_true)
    pred_labels = np.unique(labels_pred)
    
    # Create contingency table
    contingency_table = np.zeros((len(true_labels), len(pred_labels)), dtype=int)
    
    for i, true_label in enumerate(true_labels):
        for j, pred_label in enumerate(pred_labels):
            contingency_table[i, j] = np.sum((labels_true == true_label) & (labels_pred == pred_label))
    
    # Marginals
    sum_comb_c = sum(comb(int(n_ij), 2) for n_ij in contingency_table.flat if n_ij >= 2)
    sum_comb_k = sum(comb(int(np.sum(contingency_table[i, :])), 2) 
                     for i in range(len(true_labels)) 
                     if np.sum(contingency_table[i, :]) >= 2)
    sum_comb_c_pred = sum(comb(int(np.sum(contingency_table[:, j])), 2) 
                         for j in range(len(pred_labels)) 
                         if np.sum(contingency_table[:, j]) >= 2)
    
    n = len(labels_true)
    expected_index = sum_comb_k * sum_comb_c_pred / comb(n, 2) if n >= 2 else 0
    max_index = (sum_comb_k + sum_comb_c_pred) / 2
    
    if max_index - expected_index == 0:
        return 0.0
    
    return (sum_comb_c - expected_index) / (max_index - expected_index)

class KMeansClustering:
    ''' K-Means Clustering Algorithm implemented with NumPy '''
    def __init__(self, K=3, plot_steps=False, max_iters=300, init='random'):
        self.K = K
        self.centroids = None
        self.plot_steps = plot_steps
        self.max_iters = max_iters
        self.inertia_ = None
        self.init = init  # 'random' or 'kmeans++'
        self.clusters = [[] for _ in range(self.K)] # List of K lists to hold the data points assigned to each cluster 

        self.centroids = [] # List to hold the centroid positions

    def predict(self, X):
        # Predict the closest cluster each data point in X belongs to
        self.X = X
        self.n_samples, self.n_features = X.shape 

        # Initialization
        if self.init == 'kmeans++':
            self.centroids = self._init_kmeans_plus_plus(X)
        else:
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

        self.inertia_ = self._compute_inertia(self.clusters, self.centroids)
        
        return self._get_cluster_labels(self.clusters)
    
    def _init_kmeans_plus_plus(self, X):
        """K-Means++ smart initialization"""
        n_samples, _ = X.shape
        centroids = np.zeros((self.K, X.shape[1]))
            
        # 1. Choose the first centroid randomly
        first_idx = np.random.randint(n_samples)
        centroids[0] = X[first_idx]
            
        # 2. Choose the remaining centroids
        for k in range(1, self.K):
            dist_sq = np.array([min([euclidean_distance(c, x)**2 for c in centroids[:k]]) for x in X])
            probs = dist_sq / dist_sq.sum()
            cumulative_probs = probs.cumsum()
            r = np.random.rand()
                
            for j, p in enumerate(cumulative_probs):
                if r < p:
                    centroids[k] = X[j]
                    break
        return list(centroids)


    def _compute_inertia(self, clusters, centroids):
        # Compute the inertia (sum of squared distances to closest centroid)
        inertia = 0
        for cluster_idx, cluster in enumerate(clusters):
            for data_point in cluster:
                inertia += euclidean_distance(self.X[data_point], centroids[cluster_idx]) ** 2
        return inertia

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
        
    def evaluate(self, labels_true, labels_pred=None):
        """
        Evaluate clustering performance using Rand Index and Adjusted Rand Index.
        
        Parameters:
        -----------
        labels_true : array-like, shape = [n_samples]
            Ground truth class labels
        labels_pred : array-like, shape = [n_samples], optional
            Predicted cluster labels. If None, uses the last clustering result.
            
        Returns:
        --------
        dict : Dictionary containing 'rand_index' and 'adjusted_rand_index'
        """
        if labels_pred is None:
            if not hasattr(self, 'clusters') or not self.clusters:
                raise ValueError("No clustering results available. Run predict() first or provide labels_pred.")
            labels_pred = self._get_cluster_labels(self.clusters)
        
        ri = rand_index(labels_true, labels_pred)
        ari = adjusted_rand_index(labels_true, labels_pred)
        
        return {
            'rand_index': ri,
            'adjusted_rand_index': ari
        }
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