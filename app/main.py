# main.py
import streamlit as st
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

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

# Import Classification
from src.classification import random_forest_classification
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# Set paths
DATA_DIR = "E:\\web-data-mining\\data"
RESULTS_DIR = "E:\\web-data-mining\\results"
preprocessed_path = "E:\\web-data-mining\\results\\preprocessed_data_concise.csv"

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
    "Run Classification",
    "Run Learning",
    "Run Mining",
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

# Step: Preprocess Data
elif step == "Preprocess Data":
    st.subheader("🛠️ Data Preprocessing")

    if os.path.exists(preprocessed_path):
        st.success("✅ Preprocessed data found. Loading from file...")
        try:
            # Load the preprocessed data from CSV
            st.session_state.processed_data = pd.read_csv(preprocessed_path)
            st.dataframe(st.session_state.processed_data.head())
        except Exception as e:
            st.error(f"❌ Failed to load preprocessed data: {e}")
    else:
        st.warning("⚠️ Preprocessed data file not found.")

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
        st.warning("⚠️ Please preprocess data first.")


# Step: Clustering
elif step == "Run Clustering":
    if st.session_state.processed_data is not None:
        prior_data = st.session_state.processed_data

        if len(prior_data) < 5:
            st.warning(f"⚠️ Not enough data to perform clustering (need ≥5 rows, got {len(prior_data)})")
        else:
            # Sidebar controls
            st.sidebar.subheader("KMeans Parameters")
            n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)

            st.sidebar.subheader("DBSCAN Parameters")
            eps = st.sidebar.slider("Epsilon", 0.1, 1.0, 0.5)
            min_samples = st.sidebar.slider("Min Samples", 1, 10, 5)

            # Session state init
            for key in ['show_kmeans_result', 'show_kmeans_pca', 'show_dbscan_result', 'show_dbscan_pca']:
                if key not in st.session_state:
                    st.session_state[key] = False

            # KMeans Clustering Button
            if st.button("🧠 Run KMeans Clustering"):
                clustered_kmeans, model_kmeans = kmeans_clustering(prior_data.copy(), n_clusters=n_clusters, results_dir=RESULTS_DIR)
                st.session_state.clustered_kmeans = clustered_kmeans
                st.session_state.model_kmeans = model_kmeans
                st.session_state.show_kmeans_result = True

            # Show KMeans PCA Plot
            if st.button("📉 Show KMeans PCA Cluster Plot"):
                if 'clustered_kmeans' in st.session_state:
                    plot_clusters(st.session_state.clustered_kmeans, kmeans_model=st.session_state.model_kmeans, results_dir=RESULTS_DIR)
                    st.session_state.show_kmeans_pca = True

            # DBSCAN Clustering Button
            if st.button("🧠 Run DBSCAN Clustering"):
                clustered_dbscan, model_dbscan = dbscan_clustering(prior_data.copy(), eps=eps, min_samples=min_samples, results_dir=RESULTS_DIR)
                st.session_state.clustered_dbscan = clustered_dbscan
                st.session_state.model_dbscan = model_dbscan
                st.session_state.show_dbscan_result = True

            # Show DBSCAN PCA Plot
            if st.button("📉 Show DBSCAN PCA Cluster Plot"):
                if 'clustered_dbscan' in st.session_state:
                    plot_clusters(st.session_state.clustered_dbscan, dbscan_model=st.session_state.model_dbscan, results_dir=RESULTS_DIR)
                    st.session_state.show_dbscan_pca = True

            # ======================
            # Show Results on Page
            # ======================
            if st.session_state.show_kmeans_result:
                st.subheader("🔍 KMeans Clustering Result")
                st.dataframe(st.session_state.clustered_kmeans[['user_id', 'kmeans_cluster']].drop_duplicates().head())

            if st.session_state.show_kmeans_pca:
                st.subheader("📊 KMeans PCA Cluster Plot")
                st.image(os.path.join(RESULTS_DIR, 'kmeans_pca_clustering_plot.png'))

            if st.session_state.show_dbscan_result:
                st.subheader("🔍 DBSCAN Clustering Result")
                st.dataframe(st.session_state.clustered_dbscan[['user_id', 'dbscan_cluster']].drop_duplicates().head())

            if st.session_state.show_dbscan_pca:
                st.subheader("📊 DBSCAN PCA Cluster Plot")
                st.image(os.path.join(RESULTS_DIR, 'dbscan_pca_clustering_plot.png'))

    else:
        st.warning("⚠️ Please preprocess data first.")


# Step: Classification
elif step == "Run Classification":
    if st.session_state.processed_data is not None:
        st.subheader("🔍 Classification Analysis")
        prior_data = st.session_state.processed_data

        if len(prior_data) < 5:
            st.warning(f"⚠️ Not enough data to perform classification (need ≥5 rows, got {len(prior_data)})")
        else:
            # Sidebar controls for features and target column
            target_column = st.sidebar.text_input("Target Column", "reordered")
            features = st.sidebar.text_area("Features (comma-separated)", "total_orders, avg_days_between_orders, reorder_ratio, avg_basket_size")

            # Button to load existing classification results
            if st.button("📂 Load Classification Results"):
                model_path = os.path.join(RESULTS_DIR, "random_forest_model.pkl")
                cm_plot_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
                roc_auc_plot_path = os.path.join(RESULTS_DIR, "roc_auc_curve.png")

                try:
                    # Load the entire model and results (assumes it was saved as a dictionary)
                    loaded_model_data = joblib.load(model_path)

                    # Check that model and predictions are loaded correctly
                    rf_classifier = loaded_model_data['model']
                    y_test = loaded_model_data['y_test']
                    y_pred = loaded_model_data['y_pred']
                    accuracy = loaded_model_data['accuracy']
                    precision = loaded_model_data['precision']
                    recall = loaded_model_data['recall']
                    f1 = loaded_model_data['f1']
                    execution_time = loaded_model_data['execution_time']
                    roc_auc = loaded_model_data['roc_auc'] if 'roc_auc' in loaded_model_data else None

                    # Display the classification report and results
                    st.subheader("📊 Classification Report (Table Format)")

                    # Convert classification report to a dict, then to a DataFrame
                    report_dict = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report_dict).transpose()

                    # Display as a Streamlit table
                    st.dataframe(report_df.style.format("{:.4f}"))

                    # Display summary metrics in another table
                    summary_df = pd.DataFrame({
                        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "Execution Time (s)"],
                        "Value": [accuracy, precision, recall, f1, execution_time]
                    })
                    st.subheader("📌 Summary Metrics")
                    st.table(summary_df.style.format({"Value": "{:.4f}"}))

                    if os.path.exists(cm_plot_path):
                        st.subheader("📉 Confusion Matrix")
                        st.image(cm_plot_path)

                    if roc_auc is not None and os.path.exists(roc_auc_plot_path):
                        st.subheader("📈 ROC AUC Curve")
                        st.image(roc_auc_plot_path)

                except Exception as e:
                    st.error(f"❌ Failed to load classification results: {e}")

    else:
        st.warning("⚠️ Please preprocess data first.")


# Step: Learning
elif step == "Run Learning":
    if st.session_state.processed_data is not None:
        st.subheader("📘 Learning Models (KNN & XGBoost)")
        prior_data = st.session_state.processed_data

        # Define results directory if not already done
        results_dir = st.session_state.get("results_dir", "E:\\web-data-mining\\results")

        if st.button("📂 Load Learning Results"):
            models = ["KNN", "XGBoost"]
            for model_name in models:
                st.subheader(f"🔹 {model_name} Results")

                # Load model and metrics
                model_path = os.path.join(results_dir, f"{model_name}_model.pkl")
                if os.path.exists(model_path):
                    data = joblib.load(model_path)

                    # Summary metrics
                    summary_df = pd.DataFrame({
                        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "Execution Time (s)", "ROC-AUC"],
                        "Value": [
                            data['accuracy'],
                            data['precision'],
                            data['recall'],
                            data['f1'],
                            data['execution_time'],
                            data['roc_auc'] if data['roc_auc'] is not None else "N/A"
                        ]
                    })
                    st.table(summary_df)

                    # Confusion Matrix
                    cm_img = os.path.join(results_dir, f"{model_name}_confusion_matrix.png")
                    if os.path.exists(cm_img):
                        st.image(cm_img, caption=f"{model_name} Confusion Matrix")

                    # ROC Curve
                    roc_img = os.path.join(results_dir, f"{model_name}_roc_auc_curve.png")
                    if os.path.exists(roc_img):
                        st.image(roc_img, caption=f"{model_name} ROC-AUC Curve")

                    # Feature Importance (XGBoost only)
                    if model_name == "XGBoost":
                        feat_img = os.path.join(results_dir, f"{model_name}_feature_importance.png")
                        if os.path.exists(feat_img):
                            st.image(feat_img, caption="XGBoost Feature Importance")
                else:
                    st.warning(f"No results found for {model_name}")
    else:
        st.warning("⚠️ Please run preprocessing/classification first to generate data.")

