import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

def k_means(X, n_clusters, max_iter=100):
    np.random.seed(42)
    n_samples, n_features = X.shape
    centers = X[np.random.choice(n_samples, n_clusters, replace=False)]

    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels


online_retail = fetch_openml(data_id=352, as_frame=True)
X = online_retail.data


X_encoded = X.apply(LabelEncoder().fit_transform)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_encoded)


outlier_detector = IsolationForest(contamination=0.1)
outliers = outlier_detector.fit_predict(X_scaled)
X_cleaned = X_scaled[outliers == 1]


n_clusters = 5
centers, labels = k_means(X_cleaned, n_clusters)


plt.figure(figsize=(8, 6))
for i in range(n_clusters):
    plt.scatter(X_cleaned[labels == i, 0], X_cleaned[labels == i, 1], label=f'Cluster {i+1}')
plt.scatter(centers[:, 0], centers[:, 1], color='black', marker='x', label='Cluster Centers')
plt.title('Clustered Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()
