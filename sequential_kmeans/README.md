# Sequential K-Means Clustering with NumPy

A custom implementation of the Sequential K-Means clustering algorithm using NumPy, with comparison to scikit-learn's implementation.

## Files

- `sequential_kmeans.py` - Custom Sequential K-Means implementation using NumPy
- `main.py` - Simple example running the custom K-Means
- `tutorial.py` - Comparison between custom K-Means, scikit-learn KMeans, and MiniBatchKMeans

## Features

- Pure NumPy implementation of K-Means clustering
- Expectation-Maximization (EM) algorithm
- Convergence detection
- Visual comparison with scikit-learn implementations

## Results

* **TEST 1: Sequential K-Means**
    The default random initialization run correctly identified the four clusters. The visual result shows well-defined groups and centroids precisely located at the center of each data cluster.

* **TEST 2: Sequential K-Means with kmeans++**
    Running with the **K-Means++** initialization produced an **inferior result**. The algorithm incorrectly merged two distinct clusters and positioned a centroid in an empty area. This occurrence demonstrates the heuristic nature of K-Means and that, although K-Means++ is generally more robust, it does not guarantee the globally optimal solution in all runs.

* **TEST 3: Sequential K-Means with elbow**
    The inertia analysis (WCSS) showed a sharp drop until K=4, at which point the curve stabilizes, validating the choice of 4 clusters for the tests
    