import numpy as np
from kmeans import KMeansClustering
        
from sklearn.datasets import make_blobs

np.random.seed(42)

batch_size = 45
centers = np.array([[1, 1], [-1, -1], [1, -1]])
clusters = len(centers)
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7) # type: ignore

kmeans = KMeansClustering(K=clusters, max_iters=200, plot_steps=True)
y_pred = kmeans.predict(X)

