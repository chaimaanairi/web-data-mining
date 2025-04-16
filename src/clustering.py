#KMeans & DBSCAN clustering
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from src.data_loader import load_data
from src.data_preprocessing import preprocess_data
from sklearn.decomposition import PCA

def kmeans_clustering(prior_data, n_clusters=5):
    # Select relevant features for clustering
    features = ['total_orders', 'avg_days_between_orders', 'reorder_ratio', 'avg_basket_size']

    # Extract features for clustering
    X = prior_data[features]

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit the KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)

    # Assign cluster labels to the data
    prior_data['cluster'] = kmeans.labels_

    # Elbow Method for optimal clusters (Optional step)
    # Calculate distortion (inertia) for each number of clusters
    distortions = []
    for i in range(1, 11):
        kmeans_temp = KMeans(n_clusters=i, random_state=42)
        kmeans_temp.fit(X_scaled)
        distortions.append(kmeans_temp.inertia_)

    # Plot the elbow graph
    plt.figure(figsize=(8,6))
    plt.plot(range(1, 11), distortions, marker='o')
    plt.title("Elbow Method for Optimal Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Distortion")
    plt.show()

    return prior_data, kmeans

# Load and preprocess data
data = load_data("../data")
prior_data = preprocess_data(data)

# Run KMeans clustering
clustered_data, kmeans_model = kmeans_clustering(prior_data, n_clusters=5)

# Preview the clustered data
print(clustered_data[['user_id', 'cluster']].head())

def plot_clusters(prior_data, kmeans_model):
    # Perform PCA to reduce data to 2D for visualization
    pca = PCA(n_components=2)
    X_scaled = StandardScaler().fit_transform(prior_data[['total_orders', 'avg_days_between_orders', 'reorder_ratio', 'avg_basket_size']])
    pca_result = pca.fit_transform(X_scaled)

    # Add PCA results to data
    prior_data['PCA1'] = pca_result[:, 0]
    prior_data['PCA2'] = pca_result[:, 1]

    # Plot clusters
    plt.figure(figsize=(8,6))
    plt.scatter(prior_data['PCA1'], prior_data['PCA2'], c=prior_data['cluster'], cmap='viridis', alpha=0.6)
    plt.title('KMeans Clustering - PCA Reduction (2D)')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.colorbar(label='Cluster')
    plt.show()

# Visualize the clusters
plot_clusters(clustered_data, kmeans_model)