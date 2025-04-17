# main.py
import streamlit as st
import os
import sys
import pandas as pd

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules
from src.data_loader import load_data
from src.data_preprocessing import preprocess_data
from src.clustering import kmeans_clustering, dbscan_clustering, plot_clusters
from src.data_visualizations import (
    orders_by_day,
    orders_by_hour,
    top_10_most_ordered,
    avg_days_between_orders,
    reorder_frequency,
    most_reordered_products,
    top_departments,
    top_aisles
)

# Import Preprocessed Data Visualizations
from src.preprocessed_data_visualization import (
    plot_total_orders,
    plot_avg_days_between_orders,
    plot_reorder_ratio,
    plot_avg_basket_size,
    plot_basket_size_vs_reorder_ratio,
    plot_correlation_heatmap
)

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
            # Load the data
            st.session_state.raw_data = load_data(DATA_DIR)
            st.success("✅ Dataset loaded successfully!")

            # Visualize each table
            # Display available tables in raw_data
            for table_name, table_data in st.session_state.raw_data.items():
                st.write(f"### {table_name.capitalize()}")
                st.dataframe(table_data.head())  # Show the first few rows of each table

        except Exception as e:
            st.error(f"Failed to load data: {e}")

# Step: Raw Visualizations
elif step == "Visualize Raw Data":
    if st.session_state.raw_data is not None:
        st.subheader("📊 Raw Data Visualizations")

        st.write("### Orders by Day")
        orders_by_day(st.session_state.raw_data['orders'])

        st.write("### Orders by Hour")
        orders_by_hour(st.session_state.raw_data['orders'])

        st.write("### Top 10 Most Ordered Products")
        top_10_most_ordered(
            st.session_state.raw_data['order_products_prior'],
            st.session_state.raw_data['products']
        )

        st.write("### Average Days Between Orders")
        avg_days_between_orders(st.session_state.raw_data['orders'])

        st.write("### Reorder Frequency Distribution")
        reorder_frequency(st.session_state.raw_data['order_products_prior'])

        st.write("### Most Reordered Products")
        most_reordered_products(
            st.session_state.raw_data['order_products_prior'],
            st.session_state.raw_data['products']
        )

        st.write("### Top Departments")
        top_departments(
            st.session_state.raw_data['order_products_prior'],
            st.session_state.raw_data['products'],
            st.session_state.raw_data['departments']
        )

        st.write("### Top Aisles")
        top_aisles(
            st.session_state.raw_data['order_products_prior'],
            st.session_state.raw_data['products'],
            st.session_state.raw_data['aisles'],
            st.session_state.raw_data['departments']
        )

    else:
        st.warning("Please upload data first.")

# Step: Preprocess
elif step == "Preprocess Data":
    if st.session_state.raw_data is not None:
        st.subheader("🛠️ Data Preprocessing")

        if st.button("🔄 Preprocess Data"):
            try:
                results_dir = "E:\\web-data-mining\\results"
                os.makedirs(results_dir, exist_ok=True)
                st.session_state.processed_data = preprocess_data(st.session_state.raw_data, results_dir)
                st.success("✅ Data preprocessed successfully!")
                st.dataframe(st.session_state.processed_data.head())
            except Exception as e:
                st.error(f"Failed to preprocess data: {e}")
    else:
        st.warning("Please upload data first.")

# Step: Preprocessed Visualizations
elif step == "Visualize Preprocessed Data":
    if st.session_state.processed_data is not None:
        st.subheader("📊 Preprocessed Data Visualizations")

        # Call the functions to show preprocessed data visualizations
        st.write("### Total Orders per User")
        plot_total_orders(st.session_state.processed_data)

        st.write("### Average Days Between Orders per User")
        plot_avg_days_between_orders(st.session_state.processed_data)

        st.write("### User Reorder Ratio Distribution")
        plot_reorder_ratio(st.session_state.processed_data)

        st.write("### Average Basket Size per User")
        plot_avg_basket_size(st.session_state.processed_data)

        st.write("### Relationship Between Basket Size and Reorder Ratio")
        plot_basket_size_vs_reorder_ratio(st.session_state.processed_data)

        st.write("### Correlation Heatmap of Engineered Features")
        plot_correlation_heatmap(st.session_state.processed_data)

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
