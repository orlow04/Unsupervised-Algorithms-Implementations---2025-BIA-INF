import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sequential_kmeans import SequentialKMeans

np.random.seed(42)
batch_size = 45
centers = np.array([[1, 1], [-1, -1], [1, -1]])
n_clusters = len(centers)
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)


seq_kmeans = SequentialKMeans(K=n_clusters, max_iters=50, init='kmeans++')
t0 = time.time()
seq_kmeans.predict(X)
t_sequential = time.time() - t0

mbk = MiniBatchKMeans(
    init="k-means++",
    n_clusters=n_clusters,
    batch_size=1,
    n_init=10,
    max_no_improvement=10,
    verbose=0,
)
t0 = time.time()
mbk.fit(X)
t_mini_batch = time.time() - t0


def align_centroids(reference_centers, centers):
    order = pairwise_distances_argmin(reference_centers, centers)
    return centers[order]

reference_centers = mbk.cluster_centers_
seq_kmeans_centroids_aligned = align_centroids(reference_centers, np.asarray(seq_kmeans.centroids))

seq_kmeans_labels = pairwise_distances_argmin(X, seq_kmeans_centroids_aligned)
mbk_labels = mbk.labels_ 


fig = plt.figure(figsize=(9, 3))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
colors = ["#4EACC5", "#FF9C34", "#4E9A06"]

ax = fig.add_subplot(1, 3, 1)
for k, col in zip(range(n_clusters), colors):
    my_members = (seq_kmeans_labels == k)
    cluster_center = seq_kmeans_centroids_aligned[k]
    ax.plot(X[my_members, 0], X[my_members, 1], "w", markerfacecolor=col, marker=".")
    ax.plot(cluster_center[0], cluster_center[1], "o", markerfacecolor=col, markeredgecolor="k", markersize=6)
ax.set_title("Nosso SequentialKMeans")
ax.set_xticks(())
ax.set_yticks(())
plt.text(-3.5, 1.8, f"Tempo: {t_sequential:.2f}s\nInércia: {seq_kmeans.inertia_:.0f}")

ax = fig.add_subplot(1, 3, 2)
for k, col in zip(range(n_clusters), colors):
    my_members = (mbk_labels == k)
    cluster_center = reference_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], "w", markerfacecolor=col, marker=".")
    ax.plot(cluster_center[0], cluster_center[1], "o", markerfacecolor=col, markeredgecolor="k", markersize=6)
ax.set_title("SKlearn MiniBatchKMeans")
ax.set_xticks(())
ax.set_yticks(())
plt.text(-3.5, 1.8, f"Tempo: {t_mini_batch:.2f}s\nInércia: {mbk.inertia_:.0f}")

plt.show()