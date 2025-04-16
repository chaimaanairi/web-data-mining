import streamlit as st
import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary functions from src
from src.data_loader import load_data
from src.data_preprocessing import preprocess_data
from src.clustering import kmeans_clustering, dbscan_clustering, plot_clusters

# Set up paths
DATA_DIR = "E:\\web-data-mining\\data"
RESULTS_DIR = "E:\\web-data-mining\\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Page config
st.set_page_config(page_title="Customer Segmentation App", layout="wide")

# Sidebar
st.sidebar.title("🔧 Settings")
st.sidebar.markdown("Upload your dataset and select clustering parameters.")

# Main title
st.title("🛒 Customer Segmentation & Behavior Analysis")

# Upload dataset
st.sidebar.subheader("1. Upload Dataset Folder")
data_loaded = False
data = None
prior_data = None

# Simulate "loading" the dataset when the user clicks the button
if st.sidebar.button("📂 Load Sample Dataset"):
    try:
        data = load_data(DATA_DIR)
        data_loaded = True
    except Exception as e:
        st.error(f"Error loading data: {e}")
        data_loaded = False

if data_loaded:
    st.success("✅ Dataset loaded successfully!")

    if st.checkbox("Show raw `orders` data"):
        st.dataframe(data["orders"].head(10))

    # Preprocess
    st.sidebar.subheader("2. Preprocess and Analyze")
    if st.sidebar.button("⚙️ Preprocess Data"):
        with st.spinner("Preprocessing..."):
            try:
                prior_data = preprocess_data(data, results_dir=RESULTS_DIR)
                st.success("✅ Preprocessing done!")
            except Exception as e:
                st.error(f"Error during preprocessing: {e}")

        if st.checkbox("Show processed sample"):
            if prior_data is not None:
                st.dataframe(prior_data[['user_id', 'order_id', 'reorder_ratio', 'avg_basket_size']].head(10))

        # Clustering
        st.sidebar.subheader("3. Run Clustering")
        n_clusters = st.sidebar.slider("Number of Clusters (KMeans)", 2, 10, 5)
        eps = st.sidebar.slider("Epsilon (DBSCAN)", 0.1, 1.0, 0.5)
        min_samples = st.sidebar.slider("Min Samples (DBSCAN)", 1, 10, 5)

        # Check if the number of samples is sufficient for KMeans clustering
        if len(prior_data) < n_clusters:
            st.error(f"❌ Number of samples ({len(prior_data)}) is less than the number of clusters ({n_clusters}). Please reduce the number of clusters or increase the number of samples.")
        else:
            # KMeans Clustering
            if st.sidebar.button("🧠 Run KMeans Clustering"):
                with st.spinner("Clustering..."):
                    try:
                        clustered_data_kmeans, _ = kmeans_clustering(prior_data, n_clusters=n_clusters, results_dir=RESULTS_DIR)
                        st.success("✅ KMeans Clustering complete!")
                    except Exception as e:
                        st.error(f"Error during KMeans clustering: {e}")

                st.subheader("📊 KMeans Clustered Data")
                if clustered_data_kmeans is not None:
                    st.dataframe(clustered_data_kmeans[['user_id', 'kmeans_cluster']].drop_duplicates().head(20))

                st.subheader("📈 KMeans Elbow Method Plot")
                elbow_plot_path = os.path.join(RESULTS_DIR, 'elbow_method_plot.png')
                if os.path.exists(elbow_plot_path):
                    st.image(elbow_plot_path)
                else:
                    st.warning("Elbow plot not found in results directory.")

            # DBSCAN Clustering
            if st.sidebar.button("🧠 Run DBSCAN Clustering"):
                with st.spinner("Clustering..."):
                    try:
                        clustered_data_dbscan, _ = dbscan_clustering(prior_data, eps=eps, min_samples=min_samples, results_dir=RESULTS_DIR)
                        st.success("✅ DBSCAN Clustering complete!")
                    except Exception as e:
                        st.error(f"Error during DBSCAN clustering: {e}")

                st.subheader("📊 DBSCAN Clustered Data")
                if clustered_data_dbscan is not None:
                    st.dataframe(clustered_data_dbscan[['user_id', 'dbscan_cluster']].drop_duplicates().head(20))

            # Visualizations for both KMeans and DBSCAN
            if prior_data is not None:
                st.subheader("📉 PCA Visualizations")
                plot_clusters(prior_data, kmeans_model=None, dbscan_model=None, results_dir=RESULTS_DIR)

            # Export Clustered Data
            st.sidebar.subheader("4. Export Data")
            if st.sidebar.button("📤 Export KMeans Clustered Data"):
                if clustered_data_kmeans is not None:
                    export_kmeans_path = os.path.join(RESULTS_DIR, "clustered_data_kmeans.csv")
                    clustered_data_kmeans[['user_id', 'kmeans_cluster']].to_csv(export_kmeans_path, index=False)
                    st.success(f"KMeans clustered data exported to {export_kmeans_path}")
                else:
                    st.error("KMeans data not available.")

            if st.sidebar.button("📤 Export DBSCAN Clustered Data"):
                if clustered_data_dbscan is not None:
                    export_dbscan_path = os.path.join(RESULTS_DIR, "clustered_data_dbscan.csv")
                    clustered_data_dbscan[['user_id', 'dbscan_cluster']].to_csv(export_dbscan_path, index=False)
                    st.success(f"DBSCAN clustered data exported to {export_dbscan_path}")
                else:
                    st.error("DBSCAN data not available.")
else:
    st.warning("👈 Use the sidebar to load the dataset.")
