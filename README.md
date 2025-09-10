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
kmeans = KMeansClustering(k=3)
labels = kmeans.fit(X)
```

## Requirements

- numpy
- matplotlib
- scikit-learn (for comparison and data generation)
