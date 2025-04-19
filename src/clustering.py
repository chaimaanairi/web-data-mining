# KMeans & DBSCAN clustering
import os
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Clustering functions
def kmeans_clustering(prior_data, n_clusters=5, results_dir=None):
    features = ['total_orders', 'avg_days_between_orders', 'reorder_ratio', 'avg_basket_size']
    X = prior_data[features]

    if len(X) < n_clusters:
        raise ValueError(f"Number of samples ({len(X)}) is less than the number of clusters ({n_clusters}).")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)
    prior_data['kmeans_cluster'] = kmeans.labels_

    # Elbow method
    distortions = []
    for i in range(1, 11):
        kmeans_temp = KMeans(n_clusters=i, random_state=42)
        kmeans_temp.fit(X_scaled)
        distortions.append(kmeans_temp.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), distortions, marker='o')
    plt.title("Elbow Method for Optimal Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Distortion")
    elbow_plot_path = os.path.join(results_dir, 'elbow_method_plot.png')
    plt.savefig(elbow_plot_path)
    plt.close()
    print(f"Elbow plot saved to {elbow_plot_path}")

    return prior_data, kmeans

def dbscan_clustering(prior_data, eps=0.5, min_samples=5, results_dir=None):
    features = ['total_orders', 'avg_days_between_orders', 'reorder_ratio', 'avg_basket_size']
    sample_size = min(10000, len(prior_data))
    sampled_data = prior_data.sample(n=sample_size, random_state=42)
    X = sampled_data[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X_scaled)

    sampled_data['dbscan_cluster'] = dbscan.labels_

    plt.figure(figsize=(8, 6))
    plt.scatter(sampled_data['total_orders'], sampled_data['avg_basket_size'],
                c=sampled_data['dbscan_cluster'], cmap='viridis', alpha=0.6)
    plt.title('DBSCAN Clustering')
    plt.xlabel('Total Orders')
    plt.ylabel('Average Basket Size')
    plt.colorbar(label='Cluster')
    dbscan_plot_path = os.path.join(results_dir, 'dbscan_clustering_plot.png')
    plt.savefig(dbscan_plot_path)
    plt.close()
    print(f"DBSCAN plot saved to {dbscan_plot_path}")

    return sampled_data, dbscan

def plot_clusters(prior_data, kmeans_model=None, dbscan_model=None, results_dir=None):
    pca = PCA(n_components=2)
    X_scaled = StandardScaler().fit_transform(
        prior_data[['total_orders', 'avg_days_between_orders', 'reorder_ratio', 'avg_basket_size']])
    pca_result = pca.fit_transform(X_scaled)
    prior_data['PCA1'] = pca_result[:, 0]
    prior_data['PCA2'] = pca_result[:, 1]

    if kmeans_model:
        plt.figure(figsize=(8, 6))
        plt.scatter(prior_data['PCA1'], prior_data['PCA2'], c=prior_data['kmeans_cluster'], cmap='viridis', alpha=0.6)
        plt.title('KMeans Clustering - PCA Reduction (2D)')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.colorbar(label='Cluster')
        kmeans_pca_plot_path = os.path.join(results_dir, 'kmeans_pca_clustering_plot.png')
        plt.savefig(kmeans_pca_plot_path)
        plt.close()
        print(f"KMeans PCA plot saved to {kmeans_pca_plot_path}")

    if dbscan_model:
        plt.figure(figsize=(8, 6))
        plt.scatter(prior_data['PCA1'], prior_data['PCA2'], c=prior_data['dbscan_cluster'], cmap='viridis', alpha=0.6)
        plt.title('DBSCAN Clustering - PCA Reduction (2D)')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.colorbar(label='Cluster')
        dbscan_pca_plot_path = os.path.join(results_dir, 'dbscan_pca_clustering_plot.png')
        plt.savefig(dbscan_pca_plot_path)
        plt.close()
        print(f"DBSCAN PCA plot saved to {dbscan_pca_plot_path}")


# Only run this if the script is executed directly (not imported)
if __name__ == "__main__":
    from src.data_loader import load_data
    from src.data_preprocessing import preprocess_data

    data = load_data("E:\\web-data-mining\\data")
    results_dir = "E:\\web-data-mining\\results"
    os.makedirs(results_dir, exist_ok=True)

    prior_data = preprocess_data(data, results_dir)

    if len(prior_data) >= 5:
        clustered_data_kmeans, kmeans_model = kmeans_clustering(prior_data.copy(), n_clusters=5, results_dir=results_dir)

        clustered_data_dbscan, dbscan_model = dbscan_clustering(prior_data.copy(), eps=0.5, min_samples=5, results_dir=results_dir)

        # Save KMeans clustered data
        kmeans_csv_path = os.path.join(results_dir, 'clustered_data_kmeans.csv')
        clustered_data_kmeans[['user_id', 'kmeans_cluster']].to_csv(kmeans_csv_path, index=False)
        print(f"KMeans clustered data saved to {kmeans_csv_path}")

        # Save DBSCAN clustered sampled data
        dbscan_csv_path = os.path.join(results_dir, 'clustered_data_dbscan_sample.csv')
        clustered_data_dbscan[['user_id', 'dbscan_cluster']].to_csv(dbscan_csv_path, index=False)
        print(f"DBSCAN clustered sampled data saved to {dbscan_csv_path}")

        # Visualize clusters
        plot_clusters(clustered_data_kmeans, kmeans_model=kmeans_model, results_dir=results_dir)
        plot_clusters(clustered_data_dbscan, dbscan_model=dbscan_model, results_dir=results_dir)
    else:
        print(f"Not enough samples to run clustering. Only {len(prior_data)} rows available.")
