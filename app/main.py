# main.py
import streamlit as st
import os
import sys

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules
from src.data_loader import load_data
from src.data_preprocessing import preprocess_data
from src.clustering import kmeans_clustering, dbscan_clustering, plot_clusters
from src.visualizations import show_raw_data_viz, show_preprocessed_data_viz

# Set paths
DATA_DIR = "E:\\web-data-mining\\data"
RESULTS_DIR = "E:\\web-data-mining\\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Streamlit config
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🛍️ Customer Segmentation & Behavior Explorer")

# Sidebar section
st.sidebar.title("Steps")
step = st.sidebar.radio("Choose a step:", [
    "Upload Data",
    "Visualize Raw Data",
    "Preprocess Data",
    "Visualize Preprocessed Data",
    "Run Clustering",
    "Export Results"
])

# Session state for holding data
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'clustered_kmeans' not in st.session_state:
    st.session_state.clustered_kmeans = None
if 'clustered_dbscan' not in st.session_state:
    st.session_state.clustered_dbscan = None

# Step: Upload
if step == "Upload Data":
    if st.button("📂 Load Sample Dataset"):
        try:
            st.session_state.raw_data = load_data(DATA_DIR)
            st.success("✅ Dataset loaded successfully!")
            st.dataframe(st.session_state.raw_data["orders"].head())
        except Exception as e:
            st.error(f"Failed to load data: {e}")

# Step: Raw Visualizations
elif step == "Visualize Raw Data":
    if st.session_state.raw_data is not None:
        st.subheader("📊 Raw Data Visualizations")
        show_raw_data_viz(st.session_state.raw_data)
    else:
        st.warning("Please upload data first.")

# Step: Preprocess
elif step == "Preprocess Data":
    if st.session_state.raw_data is not None:
        if st.button("⚙️ Run Preprocessing"):
            with st.spinner("Processing..."):
                try:
                    processed = preprocess_data(st.session_state.raw_data, results_dir=RESULTS_DIR)
                    st.session_state.processed_data = processed
                    st.success("✅ Preprocessing complete!")
                    st.dataframe(processed.head())
                except Exception as e:
                    st.error(f"Preprocessing error: {e}")
    else:
        st.warning("Please upload data first.")

# Step: Preprocessed Visualizations
elif step == "Visualize Preprocessed Data":
    if st.session_state.processed_data is not None:
        st.subheader("📊 Preprocessed Data Visualizations")
        show_preprocessed_data_viz(st.session_state.processed_data)
    else:
        st.warning("Please preprocess data first.")

# Step: Clustering
elif step == "Run Clustering":
    if st.session_state.processed_data is not None:
        prior_data = st.session_state.processed_data

        if len(prior_data) < 5:
            st.warning(f"⚠️ Not enough data to perform clustering (need ≥5 rows, got {len(prior_data)})")
        else:
            st.sidebar.subheader("KMeans Parameters")
            n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)

            st.sidebar.subheader("DBSCAN Parameters")
            eps = st.sidebar.slider("Epsilon", 0.1, 1.0, 0.5)
            min_samples = st.sidebar.slider("Min Samples", 1, 10, 5)

            if st.button("🧠 Run KMeans Clustering"):
                clustered_kmeans, model_kmeans = kmeans_clustering(prior_data.copy(), n_clusters=n_clusters, results_dir=RESULTS_DIR)
                st.session_state.clustered_kmeans = clustered_kmeans
                st.dataframe(clustered_kmeans[['user_id', 'kmeans_cluster']].drop_duplicates().head())

            if st.button("🧠 Run DBSCAN Clustering"):
                clustered_dbscan, model_dbscan = dbscan_clustering(prior_data.copy(), eps=eps, min_samples=min_samples, results_dir=RESULTS_DIR)
                st.session_state.clustered_dbscan = clustered_dbscan
                st.dataframe(clustered_dbscan[['user_id', 'dbscan_cluster']].drop_duplicates().head())

            # Optional PCA plot
            if st.button("📉 Show PCA Cluster Plot"):
                plot_clusters(prior_data, results_dir=RESULTS_DIR)
                st.image(os.path.join(RESULTS_DIR, 'kmeans_pca_clustering_plot.png'))

    else:
        st.warning("Please preprocess data first.")

# Step: Export
elif step == "Export Results":
    if st.session_state.clustered_kmeans is not None:
        kmeans_export_path = os.path.join(RESULTS_DIR, "clustered_data_kmeans.csv")
        st.session_state.clustered_kmeans[['user_id', 'kmeans_cluster']].to_csv(kmeans_export_path, index=False)
        st.success(f"KMeans exported to: {kmeans_export_path}")

    if st.session_state.clustered_dbscan is not None:
        dbscan_export_path = os.path.join(RESULTS_DIR, "clustered_data_dbscan.csv")
        st.session_state.clustered_dbscan[['user_id', 'dbscan_cluster']].to_csv(dbscan_export_path, index=False)
        st.success(f"DBSCAN exported to: {dbscan_export_path}")
