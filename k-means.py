import numpy as np
import matplotlib.pyplot as plt

def k_means(data, n_clusters, max_iter=100, tol=1e-4):
    centers = data[np.random.choice(data.shape[0], n_clusters, replace=False)]
    for _ in range(max_iter):
        distances = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
        outputs = np.argmin(distances, axis=1)
        new_centers = np.array([data[outputs == i].mean(axis=0) if len(data[outputs == i]) > 0 else centers[i] for i in range(n_clusters)])
        
        if np.linalg.norm(new_centers - centers) < tol:
            break
        centers = new_centers
    
    return centers, outputs

def draw_clusters(data, centers, outputs):
    plt.figure(figsize=(8, 6))
    cmap = plt.cm.get_cmap('tab10')
    for i in range(centers.shape[0]):
        cluster_data = data[outputs == i]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=cmap(i), label=f'Cluster {i + 1}')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', label='Final Centers')
    plt.legend()
    plt.show()

def generate_data():
    class_1 = np.random.rand(200, 2) * 50
    class_2 = np.random.rand(200, 2) * 50 + 4
    class_3 = np.random.rand(200, 2) * 60 - 2
    data = np.vstack([class_1, class_2, class_3])
    np.random.shuffle(data)
    return data

if __name__ == "__main__":
    data = generate_data()
    centers, outputs = k_means(data, 5)
    draw_clusters(data, centers, outputs)
