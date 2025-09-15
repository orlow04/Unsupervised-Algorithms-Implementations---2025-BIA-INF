import numpy as np

from sklearn.datasets import make_blobs

np.random.seed(42)

batch_size = 45
centers = np.array([[1, 1], [-1, -1], [1, -1]])
n_clusters = len(centers)
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7, return_centers=False)   # type: ignore
import time

from sklearn.cluster import KMeans

k_means = KMeans(init="k-means++", n_clusters=3, n_init=10)
t0 = time.time()
k_means.fit(X)
t_batch = time.time() - t0


from kmeans import KMeansClustering, rand_index, adjusted_rand_index

kmeans = KMeansClustering(K=n_clusters, max_iters=300, plot_steps=False, init='random')
t0 = time.time()
y_pred = kmeans.predict(X)
t_custom = time.time() - t0

from sklearn.cluster import MiniBatchKMeans

mbk = MiniBatchKMeans(
    init="k-means++",
    n_clusters=3,
    batch_size=batch_size,
    n_init=10,
    max_no_improvement=10,
    verbose=0,
)
t0 = time.time()
mbk.fit(X)
t_mini_batch = time.time() - t0
from sklearn.metrics.pairwise import pairwise_distances_argmin

k_means_cluster_centres =  k_means.cluster_centers_
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centres)

# --- Robust comparison and visualization for all three algorithms ---
from sklearn.metrics.pairwise import pairwise_distances_argmin
import matplotlib.pyplot as plt

# Helper to align centroids and labels
def align_centroids(reference_centers, centers):
    order = pairwise_distances_argmin(reference_centers, centers)
    return centers[order], order

k_means_cluster_centres = k_means.cluster_centers_
kmeans_centroids, kmeans_order = align_centroids(k_means_cluster_centres, np.asarray(kmeans.centroids))
mbk_centroids, mbk_order = align_centroids(k_means_cluster_centres, np.asarray(mbk.cluster_centers_))

k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centres)
kmeans_labels = pairwise_distances_argmin(X, kmeans_centroids)
mbk_labels = pairwise_distances_argmin(X, mbk_centroids)

# Calculate Rand Index and Adjusted Rand Index for all algorithms
sklearn_ri = rand_index(labels_true, k_means_labels)
sklearn_ari = adjusted_rand_index(labels_true, k_means_labels)

custom_ri = rand_index(labels_true, kmeans_labels)
custom_ari = adjusted_rand_index(labels_true, kmeans_labels)

minibatch_ri = rand_index(labels_true, mbk_labels)
minibatch_ari = adjusted_rand_index(labels_true, mbk_labels)

# Print comparison results
print("\n" + "="*60)
print("RAND INDEX COMPARISON")
print("="*60)
print(f"{'Algorithm':<25} {'Time (s)':<10} {'Inertia':<12} {'Rand Index':<12} {'Adj. Rand Index':<15}")
print("-"*75)
print(f"{'Scikit-learn KMeans':<25} {t_batch:<10.3f} {k_means.inertia_:<12.2f} {sklearn_ri:<12.4f} {sklearn_ari:<15.4f}")
print(f"{'Custom KMeans':<25} {t_custom:<10.3f} {kmeans.inertia_:<12.2f} {custom_ri:<12.4f} {custom_ari:<15.4f}")
print(f"{'MiniBatch KMeans':<25} {t_mini_batch:<10.3f} {mbk.inertia_:<12.2f} {minibatch_ri:<12.4f} {minibatch_ari:<15.4f}")
print("="*60)

fig = plt.figure(figsize=(12, 3))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
colors = ["#4EACC5", "#FF9C34", "#4E9A06"]

# KMeans
ax = fig.add_subplot(1, 4, 1)
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centres[k]
    ax.plot(X[my_members, 0], X[my_members, 1], "w", markerfacecolor=col, marker=".")
    ax.plot(
        cluster_center[0],
        cluster_center[1],
        "o",
        markerfacecolor=col,
        markeredgecolor="k",
        markersize=6,
    )
ax.set_title("KMeans")
ax.set_xticks(())
ax.set_yticks(())
plt.text(-3.5, 1.8, "train time: %.2fs\ninertia: %.2f\nRI: %.3f\nARI: %.3f" % (t_batch, k_means.inertia_, sklearn_ri, sklearn_ari))

# KMeansClustering custom
ax = fig.add_subplot(1, 4, 2)
for k, col in zip(range(n_clusters), colors):
    my_members = kmeans_labels == k
    cluster_center = kmeans_centroids[k]
    ax.plot(X[my_members, 0], X[my_members, 1], "w", markerfacecolor=col, marker=".")
    ax.plot(
        cluster_center[0],
        cluster_center[1],
        "o",
        markerfacecolor=col,
        markeredgecolor="k",
        markersize=6,
    )
ax.set_title("KMeansClustering - Numpy")
ax.set_xticks(())
ax.set_yticks(())
plt.text(-3.5, 1.8, "train time: %.2fs\ninertia: %.2f\nRI: %.3f\nARI: %.3f" % (t_custom, kmeans.inertia_, custom_ri, custom_ari))

# MiniBatchKMeans
ax = fig.add_subplot(1, 4, 3)
for k, col in zip(range(n_clusters), colors):
    my_members = mbk_labels == k
    cluster_center = mbk_centroids[k]
    ax.plot(X[my_members, 0], X[my_members, 1], "w", markerfacecolor=col, marker=".")
    ax.plot(
        cluster_center[0],
        cluster_center[1],
        "o",
        markerfacecolor=col,
        markeredgecolor="k",
        markersize=6,
    )
ax.set_title("MiniBatchKMeans")
ax.set_xticks(())
ax.set_yticks(())
plt.text(-3.5, 1.8, "train time: %.2fs\ninertia: %.2f\nRI: %.3f\nARI: %.3f" % (t_mini_batch, mbk.inertia_, minibatch_ri, minibatch_ari))

# Difference plot (between kmeans and sklearn)
ax = fig.add_subplot(1, 4, 4)
different = kmeans_labels != k_means_labels
identical = ~different
ax.plot(X[identical, 0], X[identical, 1], "w", markerfacecolor="#bbbbbb", marker=".")
ax.plot(X[different, 0], X[different, 1], "w", markerfacecolor="m", marker=".")
ax.set_title("Difference (custom vs sklearn)")
ax.set_xticks(())
ax.set_yticks(())

plt.show()

fig.savefig("kmeans_comparison_with_rand_index.png", dpi=300)
