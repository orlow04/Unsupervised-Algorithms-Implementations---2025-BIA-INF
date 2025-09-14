import numpy as np
from sklearn.datasets import make_blobs
from sequential_kmeans import SequentialKMeans, run_elbow_method

np.random.seed(42)

n_samples = 500
n_features = 2
centers = 4
cluster_std = 0.8

X, y_true = make_blobs(n_samples=n_samples, 
                        n_features=n_features, 
                        centers=centers, 
                        cluster_std=cluster_std, 
                        random_state=42)

print("--- Testando a implementação do SequentialKMeans Avançado ---")
print(f"Dados gerados com {n_samples} amostras e {centers} clusters verdadeiros.\n")

print(">>> INICIANDO TESTE 1: Sequential K-Means Padrão")
kmeans_padrao = SequentialKMeans(K=centers, max_iters=50, init='random')
labels_padrao = kmeans_padrao.predict(X)
print(f"Execução padrão finalizada. Inércia (WCSS): {kmeans_padrao.inertia_:.2f}")
print("Plotando o resultado do K-Means Padrão...")
kmeans_padrao.plot()

print("\n>>> INICIANDO TESTE 2: Sequential K-Means com Inicialização K-Means++")
kmeans_pp = SequentialKMeans(K=centers, max_iters=50, init='kmeans++')
labels_pp = kmeans_pp.predict(X)
print(f"Execução com K-Means++ finalizada. Inércia (WCSS): {kmeans_pp.inertia_:.2f}")
print("Plotando o resultado do K-Means++...")
kmeans_pp.plot()

print("\n>>> INICIANDO TESTE 3: Método elbow para encontrar o K ótimo")
run_elbow_method(X, max_k=10, max_iters=50, init='kmeans++')
