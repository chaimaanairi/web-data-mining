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
    "Run Clustering Models",
    "Run Classification Model",
    "Run Learning Models",
    "Run Apriori Mining",
    "Evaluation Models Performance"
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
elif step == "Run Clustering Models":
    if st.session_state.processed_data is not None:
        st.subheader("Clustering Models: KMeans & DBSCAN")
        prior_data = st.session_state.processed_data

        if len(prior_data) < 5:
            st.warning(f"⚠️ Not enough data to perform clustering (need ≥5 rows, got {len(prior_data)})")
        else:
            # Sidebar controls for KMeans parameters
            st.sidebar.subheader("KMeans Parameters")
            n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)

            # Sidebar controls for DBSCAN parameters
            st.sidebar.subheader("DBSCAN Parameters")
            eps = st.sidebar.slider("Epsilon", 0.1, 1.0, 0.5)
            min_samples = st.sidebar.slider("Min Samples", 1, 10, 5)

            # Initialize session state flags for storing the results
            for key in ['show_kmeans_result', 'show_kmeans_pca', 'show_dbscan_result', 'show_dbscan_pca']:
                if key not in st.session_state:
                    st.session_state[key] = False

            # Run KMeans clustering
            if st.button("🧠 Run KMeans Clustering"):
                clustered_kmeans, model_kmeans, kmeans_silhouette, kmeans_davies = kmeans_clustering(
                    prior_data.copy(), n_clusters=n_clusters, results_dir=RESULTS_DIR)
                st.session_state.clustered_kmeans = clustered_kmeans
                st.session_state.model_kmeans = model_kmeans
                st.session_state.kmeans_silhouette = kmeans_silhouette
                st.session_state.kmeans_davies = kmeans_davies
                st.session_state.show_kmeans_result = True

            # Show KMeans PCA plot
            if st.button("📉 Show KMeans PCA Cluster Plot"):
                if 'clustered_kmeans' in st.session_state:
                    plot_clusters(st.session_state.clustered_kmeans, kmeans_model=st.session_state.model_kmeans,
                                  results_dir=RESULTS_DIR)
                    st.session_state.show_kmeans_pca = True

            # Run DBSCAN clustering
            if st.button("🧠 Run DBSCAN Clustering"):
                clustered_dbscan, model_dbscan, dbscan_silhouette, dbscan_davies = dbscan_clustering(
                    prior_data.copy(), eps=eps, min_samples=min_samples, results_dir=RESULTS_DIR)
                st.session_state.clustered_dbscan = clustered_dbscan
                st.session_state.model_dbscan = model_dbscan
                st.session_state.dbscan_silhouette = dbscan_silhouette
                st.session_state.dbscan_davies = dbscan_davies
                st.session_state.show_dbscan_result = True

            # Show DBSCAN PCA plot
            if st.button("📉 Show DBSCAN PCA Cluster Plot"):
                if 'clustered_dbscan' in st.session_state:
                    plot_clusters(st.session_state.clustered_dbscan, dbscan_model=st.session_state.model_dbscan,
                                  results_dir=RESULTS_DIR)
                    st.session_state.show_dbscan_pca = True

            # ======================
            # Show Results on Page
            # ======================
            if st.session_state.show_kmeans_result:
                st.subheader("🔍 KMeans Clustering Result")
                st.dataframe(st.session_state.clustered_kmeans[['user_id', 'kmeans_cluster']].drop_duplicates().head())

                # Display KMeans Metrics as a table
                kmeans_metrics = pd.DataFrame({
                    "Metric": ["Silhouette Score", "Davies-Bouldin Index"],
                    "Value": [st.session_state.kmeans_silhouette, st.session_state.kmeans_davies]
                })
                st.subheader("📊 KMeans Clustering Metrics")
                st.table(kmeans_metrics)

            if st.session_state.show_kmeans_pca:
                st.subheader("📊 KMeans PCA Cluster Plot")
                st.image(os.path.join(RESULTS_DIR, 'kmeans_pca_clustering_plot.png'))

            if st.session_state.show_dbscan_result:
                st.subheader("🔍 DBSCAN Clustering Result")
                st.dataframe(st.session_state.clustered_dbscan[['user_id', 'dbscan_cluster']].drop_duplicates().head())

                # Display DBSCAN Metrics as a table
                dbscan_metrics = pd.DataFrame({
                    "Metric": ["Silhouette Score", "Davies-Bouldin Index"],
                    "Value": [
                        st.session_state.dbscan_silhouette if st.session_state.dbscan_silhouette != -1 else "Not applicable",
                        st.session_state.dbscan_davies if st.session_state.dbscan_davies != -1 else "Not applicable"
                    ]
                })
                st.subheader("📊 DBSCAN Clustering Metrics")
                st.table(dbscan_metrics)

            if st.session_state.show_dbscan_pca:
                st.subheader("📊 DBSCAN PCA Cluster Plot")
                st.image(os.path.join(RESULTS_DIR, 'dbscan_pca_clustering_plot.png'))

    else:
        st.warning("⚠️ Please preprocess data first.")


# Step: Classification
elif step == "Run Classification Model":
    if st.session_state.processed_data is not None:
        st.subheader("Classification Model: Random Forest")
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
elif step == "Run Learning Models":
    if st.session_state.processed_data is not None:
        st.subheader("Learning Models (KNN & XGBoost)")
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

# Step: Apriori Mining
elif step == "Run Apriori Mining":
    if st.session_state.processed_data is not None:
        st.subheader("Apriori Mining")
        prior_data = st.session_state.processed_data

        if st.button("📂 Load Apriori Mining Results"):
            apriori_model_path = os.path.join(RESULTS_DIR, "apriori_rules.csv")
            if os.path.exists(apriori_model_path):
                # Load rules and metrics
                rules = pd.read_csv(apriori_model_path)

                # Display metrics
                metrics = {
                    'Number of Rules': len(rules),
                    'Average Support': rules['support'].mean(),
                    'Average Confidence': rules['confidence'].mean(),
                    'Average Lift': rules['lift'].mean(),
                    'Lift Coverage (%)': (rules['lift'] >= 1.5).mean() * 100
                }

                st.write("### Apriori Mining Metrics")
                st.table(pd.DataFrame(metrics, index=[0]))

                # Show top 5 rules
                st.write("### Top 5 Rules")
                st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

                # Display Support vs Confidence plot
                support_confidence_img = os.path.join(RESULTS_DIR, "support_vs_confidence.png")
                if os.path.exists(support_confidence_img):
                    st.image(support_confidence_img, caption="Support vs Confidence (Colored by Lift)")

                # Display Lift Histogram
                lift_hist_img = os.path.join(RESULTS_DIR, "lift_histogram.png")
                if os.path.exists(lift_hist_img):
                    st.image(lift_hist_img, caption="Distribution of Lift")

            else:
                st.warning("No Apriori mining results found.")
    else:
        st.warning("⚠️ Please run preprocessing first to generate data.")



# Step: Evaluation Models Performance
elif step == "Evaluation Models Performance":
    st.subheader("📊 Evaluation of Models Performance")

    st.write("This section will provide an overview of the performance of the various models used in this project.")

    # 1. Random Forest Classification Evaluation
    if 'random_forest_model' in st.session_state:
        st.write("### Random Forest Model Evaluation")
        try:
            rf_metrics = st.session_state.get('rf_metrics', {})
            if rf_metrics:
                rf_df = pd.DataFrame(rf_metrics, index=[0])
                st.table(rf_df)
                st.image(os.path.join(RESULTS_DIR, 'random_forest_confusion_matrix.png'),
                         caption="Random Forest Confusion Matrix")
                st.image(os.path.join(RESULTS_DIR, 'random_forest_roc_auc_curve.png'),
                         caption="Random Forest ROC-AUC Curve")
            else:
                st.warning("Random Forest model evaluation metrics not available.")
        except Exception as e:
            st.error(f"❌ Failed to load Random Forest evaluation: {e}")

    # 2. KNN Classification Evaluation
    if 'knn_model' in st.session_state:
        st.write("### KNN Model Evaluation")
        try:
            knn_metrics = st.session_state.get('knn_metrics', {})
            if knn_metrics:
                knn_df = pd.DataFrame(knn_metrics, index=[0])
                st.table(knn_df)
                st.image(os.path.join(RESULTS_DIR, 'knn_confusion_matrix.png'), caption="KNN Confusion Matrix")
                st.image(os.path.join(RESULTS_DIR, 'knn_roc_auc_curve.png'), caption="KNN ROC-AUC Curve")
            else:
                st.warning("KNN model evaluation metrics not available.")
        except Exception as e:
            st.error(f"❌ Failed to load KNN evaluation: {e}")

    # 3. XGBoost Classification Evaluation
    if 'xgboost_model' in st.session_state:
        st.write("### XGBoost Model Evaluation")
        try:
            xgb_metrics = st.session_state.get('xgboost_metrics', {})
            if xgb_metrics:
                xgb_df = pd.DataFrame(xgb_metrics, index=[0])
                st.table(xgb_df)
                st.image(os.path.join(RESULTS_DIR, 'xgboost_confusion_matrix.png'), caption="XGBoost Confusion Matrix")
                st.image(os.path.join(RESULTS_DIR, 'xgboost_roc_auc_curve.png'), caption="XGBoost ROC-AUC Curve")
            else:
                st.warning("XGBoost model evaluation metrics not available.")
        except Exception as e:
            st.error(f"❌ Failed to load XGBoost evaluation: {e}")

    # 4. KMeans Clustering Evaluation
    if 'clustered_kmeans' in st.session_state:
        st.write("### KMeans Clustering Evaluation")
        try:
            # Assuming silhouette score is available from KMeans
            silhouette_score = st.session_state.get('kmeans_silhouette_score', None)
            if silhouette_score is not None:
                st.write(f"**Silhouette Score**: {silhouette_score:.4f}")
            else:
                st.warning("Silhouette score for KMeans not available.")

            # Display cluster size (number of points in each cluster)
            if 'clustered_kmeans' in st.session_state:
                cluster_sizes = st.session_state.clustered_kmeans['kmeans_cluster'].value_counts()
                st.write("**Cluster Sizes**:")
                st.write(cluster_sizes)
        except Exception as e:
            st.error(f"❌ Failed to load KMeans evaluation: {e}")

    # 5. DBSCAN Clustering Evaluation
    if 'clustered_dbscan' in st.session_state:
        st.write("### DBSCAN Clustering Evaluation")
        try:
            # Assuming silhouette score is available from DBSCAN
            silhouette_score = st.session_state.get('dbscan_silhouette_score', None)
            if silhouette_score is not None:
                st.write(f"**Silhouette Score**: {silhouette_score:.4f}")
            else:
                st.warning("Silhouette score for DBSCAN not available.")

            # Display cluster size (number of points in each cluster)
            if 'clustered_dbscan' in st.session_state:
                cluster_sizes = st.session_state.clustered_dbscan['dbscan_cluster'].value_counts()
                st.write("**Cluster Sizes**:")
                st.write(cluster_sizes)
        except Exception as e:
            st.error(f"❌ Failed to load DBSCAN evaluation: {e}")

    # 6. Apriori Mining Evaluation
    if 'apriori_rules' in st.session_state:
        st.write("### Apriori Mining Evaluation")
        try:
            apriori_rules = st.session_state.get('apriori_rules', pd.DataFrame())
            if not apriori_rules.empty:
                st.write("**Number of Rules Generated**: ", len(apriori_rules))
                st.write("**Average Support**: ", apriori_rules['support'].mean())
                st.write("**Average Confidence**: ", apriori_rules['confidence'].mean())
                st.write("**Average Lift**: ", apriori_rules['lift'].mean())

                # Lift Coverage: Percentage of rules with lift >= 1.5
                lift_coverage = (apriori_rules['lift'] >= 1.5).mean() * 100
                st.write(f"**Lift Coverage (Lift >= 1.5)**: {lift_coverage:.2f}%")

                # Show the first 5 rules
                st.write("### Top 5 Rules:")
                st.dataframe(apriori_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

                # Visualizations for Apriori
                support_confidence_plot = os.path.join(RESULTS_DIR, "support_vs_confidence.png")
                if os.path.exists(support_confidence_plot):
                    st.image(support_confidence_plot, caption="Support vs Confidence Plot")

                lift_histogram_plot = os.path.join(RESULTS_DIR, "lift_histogram.png")
                if os.path.exists(lift_histogram_plot):
                    st.image(lift_histogram_plot, caption="Lift Histogram")
            else:
                st.warning("Apriori mining rules not available.")
        except Exception as e:
            st.error(f"❌ Failed to load Apriori evaluation: {e}")

