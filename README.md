# K-Means Clustering with NumPy

A custom implementation of the K-Means clustering algorithm using NumPy, with comparison to scikit-learn's implementation.

## Files

- `kmeans.py` - Custom K-Means implementation using NumPy
- `main.py` - Simple example running the custom K-Means
- `tutorial.py` - Comparison between custom K-Means, scikit-learn KMeans, and MiniBatchKMeans

## Features

- Pure NumPy implementation of K-Means clustering
- Expectation-Maximization (EM) algorithm
- Convergence detection
- Visual comparison with scikit-learn implementations

## Usage

```python
from kmeans import KMeansClustering

# Create and fit the model
kmeans = KMeansClustering(K=3, max_iters=100)
labels = kmeans.predict(X)
```

## Requirements

- numpy
- matplotlib
- scikit-learn (for comparison and data generation)

## References
- Deep Learning Foundations and Practice(https://www.bishopbook.com/)
- https://www.youtube.com/watch?v=6UF5Ysk_2gk
- https://www.youtube.com/watch?v=5w5iUbTlpMQ&t=12s
- Probabilistic Machine Learning An Introduction (Kevin Murphy)
- Hubert, L., & Arabie, P. (1985). Comparing partitions. Journal of classification, 2(1), 193-218.
