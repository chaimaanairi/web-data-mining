import streamlit as st
import pandas as pd
from src.data_loader import load_data
from src.data_preprocessing import preprocess_data
from src.clustering import kmeans_clustering, plot_clusters

# Title of the app
st.title("Customer Segmentation and Behavior Analysis")

# Upload CSV
uploaded_file = st.file_uploader("Upload Your Dataset", type=["csv"])

if uploaded_file is not None:
    # Load data
    data = load_data(uploaded_file)

    # Preprocess the data
    prior_data = preprocess_data(data)

    # Show data preview
    st.subheader("Data Preview")
    st.write(prior_data.head())

    # Select the algorithm (KMeans for now)
    algorithm = st.selectbox("Select Algorithm", ["KMeans Clustering"])

    if algorithm == "KMeans Clustering":
        # Choose number of clusters
        n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=5)

        # Run KMeans clustering
        clustered_data, kmeans_model = kmeans_clustering(prior_data, n_clusters)

        # Display clustered data
        st.subheader(f"Clustered Data (n_clusters={n_clusters})")
        st.write(clustered_data[['user_id', 'cluster']])

        # Plot Clusters
        st.subheader("Cluster Visualization (PCA Reduction)")
        plot_clusters(clustered_data, kmeans_model)
