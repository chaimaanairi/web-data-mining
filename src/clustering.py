import os
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import time
from sklearn.utils import resample


def kmeans_clustering(prior_data, n_clusters=5, results_dir=None):
    features = ['total_orders', 'avg_days_between_orders', 'reorder_ratio', 'avg_basket_size']
    X = prior_data[features]

    if len(X) < n_clusters:
        raise ValueError(f"Number of samples ({len(X)}) is less than the number of clusters ({n_clusters}).")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000, random_state=42)
    kmeans.fit(X_scaled)
    prior_data['kmeans_cluster'] = kmeans.labels_

    # Evaluate clustering with sampling
    sample_size = min(1000, len(X_scaled))
    X_sample, labels_sample = resample(X_scaled, kmeans.labels_, n_samples=sample_size, random_state=42)

    start = time.time()
    silhouette = silhouette_score(X_sample, labels_sample)
    davies = davies_bouldin_score(X_sample, labels_sample)
    print(f"Silhouette score (sampled): {silhouette:.4f}")
    print(f"Davies-Bouldin score (sampled): {davies:.4f}")
    print(f"Evaluation took {time.time() - start:.2f} seconds")

    # Save KMeans metrics to file
    if results_dir:
        kmeans_report_path = os.path.join(results_dir, 'kmeans_report.txt')
        with open(kmeans_report_path, 'w') as f:
            f.write("KMeans Clustering Report\n")
            f.write("=========================\n")
            f.write(f"Number of clusters: {n_clusters}\n")
            f.write(f"Silhouette Score (sampled): {silhouette:.4f}\n")
            f.write(f"Davies-Bouldin Index (sampled): {davies:.4f}\n")
        print(f"KMeans report saved to {kmeans_report_path}")

    # Elbow plot using a sample
    distortions = []
    sample_X = X_scaled if len(X_scaled) < 10000 else X_scaled[:10000]
    for i in range(1, 11):
        kmeans_temp = KMeans(n_clusters=i, random_state=42)
        kmeans_temp.fit(sample_X)
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

    return prior_data, kmeans, silhouette, davies


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

    # Evaluate DBSCAN if valid
    if len(set(dbscan.labels_)) > 1 and -1 not in dbscan.labels_:
        silhouette = silhouette_score(X_scaled, dbscan.labels_)
        davies = davies_bouldin_score(X_scaled, dbscan.labels_)
    else:
        silhouette = -1
        davies = -1

    # Save DBSCAN metrics to file
    if results_dir:
        dbscan_report_path = os.path.join(results_dir, 'dbscan_report.txt')
        with open(dbscan_report_path, 'w') as f:
            f.write("DBSCAN Clustering Report\n")
            f.write("========================\n")
            f.write(f"EPS: {eps}\n")
            f.write(f"Min Samples: {min_samples}\n")
            f.write(f"Number of clusters (excluding noise): {len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)}\n")
            f.write(f"Number of noise points: {(dbscan.labels_ == -1).sum()}\n")
            f.write(f"Silhouette Score: {silhouette if silhouette != -1 else 'Not applicable'}\n")
            f.write(f"Davies-Bouldin Index: {davies if davies != -1 else 'Not applicable'}\n")
        print(f"DBSCAN report saved to {dbscan_report_path}")

    # Save cluster plot
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

    return sampled_data, dbscan, silhouette, davies  # Return 4 values



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


if __name__ == "__main__":
    preprocessed_data_path = "E:\\web-data-mining\\results\\preprocessed_data_concise.csv"
    prior_data = pd.read_csv(preprocessed_data_path)
    results_dir = "E:\\web-data-mining\\results"
    os.makedirs(results_dir, exist_ok=True)

    if len(prior_data) >= 5:
        clustered_data_kmeans, kmeans_model, silhouette, davies = kmeans_clustering(
            prior_data.copy(), n_clusters=5, results_dir=results_dir)

        clustered_data_dbscan, dbscan_model, dbscan_silhouette, dbscan_davies = dbscan_clustering(
            prior_data.copy(), eps=0.5, min_samples=5, results_dir=results_dir)

        # Save results
        clustered_data_kmeans[['user_id', 'kmeans_cluster']].to_csv(
            os.path.join(results_dir, 'clustered_data_kmeans.csv'), index=False)

        clustered_data_dbscan[['user_id', 'dbscan_cluster']].to_csv(
            os.path.join(results_dir, 'clustered_data_dbscan_sample.csv'), index=False)

        plot_clusters(clustered_data_kmeans, kmeans_model=kmeans_model, results_dir=results_dir)
        plot_clusters(clustered_data_dbscan, dbscan_model=dbscan_model, results_dir=results_dir)
    else:
        print(f"Not enough samples to run clustering. Only {len(prior_data)} rows available.")