import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class SequentialKMeans:
    ''' Sequential K-Means com inicialização K-Means++ opcional. '''
    def __init__(self, K=3, plot_steps=False, max_iters=100, init='random'):
        self.K = K
        self.plot_steps = plot_steps
        self.max_iters = max_iters
        self.init = init  # Pode ser 'random' ou 'kmeans++'
        self.centroids = []
        self.cluster_counts = []
        self.inertia_ = None # Para armazenar o valor da inércia

    def _init_kmeans_plus_plus(self, X):
        """Inicialização inteligente K-Means++"""
        n_samples, _ = X.shape
        centroids = np.zeros((self.K, X.shape[1]))
        
        # 1. Escolhe o primeiro centroide aleatoriamente
        first_idx = np.random.randint(n_samples)
        centroids[0] = X[first_idx]
        
        # 2. Escolhe os centroides restantes
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

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        if self.init == 'kmeans++':
            self.centroids = self._init_kmeans_plus_plus(X)
        else: 
            random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
            self.centroids = [self.X[idx] for idx in random_sample_idxs]

        self.cluster_counts = np.zeros(self.K, dtype=int)

        for epoch in range(self.max_iters):
            for data_point in self.X:
                closest_centroid_idx = self._closest_centroid(data_point, self.centroids)
                self.cluster_counts[closest_centroid_idx] += 1
                learning_rate = 1 / self.cluster_counts[closest_centroid_idx]
                self.centroids[closest_centroid_idx] += learning_rate * (data_point - self.centroids[closest_centroid_idx])
            
            if self.plot_steps:
                print(f"Época {epoch + 1}/{self.max_iters}")
                self.plot() 
        
        labels = self._get_final_cluster_labels(X)
        
        self._calculate_inertia(X, labels)
        
        return labels

    def _calculate_inertia(self, X, labels):
        """Calcula a inércia (WCSS - Within-Cluster Sum of Squares)"""
        inertia = 0
        for k in range(self.K):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[k])**2)
        self.inertia_ = inertia

    def _closest_centroid(self, data_point, centroids):
        distances = [euclidean_distance(data_point, c) for c in centroids]
        return np.argmin(distances)

    def _get_final_cluster_labels(self, X):
        labels = np.empty(self.n_samples, dtype=int)
        for idx, data_point in enumerate(X):
            labels[idx] = self._closest_centroid(data_point, self.centroids)
        return labels

    def plot(self):
        labels = self._get_final_cluster_labels(self.X)
        fig, ax = plt.subplots(figsize=(12, 8))
        for k in range(self.K):
            points = self.X[labels == k]
            ax.scatter(points[:, 0], points[:, 1])
        if self.centroids:
            centroid_points = np.array(self.centroids)
            ax.scatter(centroid_points[:, 0], centroid_points[:, 1], 
                       marker="x", color="black", s=150, linewidth=3)
        plt.show()

def run_elbow_method(X, max_k=10, **kwargs):
    """
    Roda o K-Means para um range de K e plota o gráfico do cotovelo.
    """
    inertias = []
    k_range = range(1, max_k + 1)
    
    print("Executando o Método  ELBOW...")
    for k in k_range:
        print(f"Testando K={k}...")
        kmeans = SequentialKMeans(K=k, **kwargs)
        kmeans.predict(X)
        inertias.append(kmeans.inertia_)
        
    # Plotando o gráfico
    plt.figure(figsize=(12, 8))
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inércia (WCSS)')
    plt.title('Método ELBOW para Encontrar o K Ótimo')
    plt.xticks(k_range)
    plt.grid(True)
    plt.show()