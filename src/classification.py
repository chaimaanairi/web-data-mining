import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, \
    roc_curve, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib


# Random Forest Classification
def random_forest_classification(data, target_column, features=None, results_dir=None):
    # Select features and target
    if features is None:
        features = ['total_orders', 'avg_days_between_orders', 'reorder_ratio', 'avg_basket_size']  # Adjust as needed
    X = data[features]
    y = data[target_column]

    # Label Encoding for target if it's categorical
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Record execution time
    start_time = time.time()

    # Train the model
    rf_classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_classifier.predict(X_test)

    # Record execution time
    end_time = time.time()
    execution_time = end_time - start_time

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    # Calculate ROC-AUC score (only for binary classification)
    roc_auc = None
    if len(set(y)) == 2:  # Check if binary classification
        roc_auc = roc_auc_score(y_test, rf_classifier.predict_proba(X_test)[:, 1])

    # Print evaluation metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Execution Time: {execution_time:.4f} seconds")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save classification report and confusion matrix
    model_results = {
        'model': rf_classifier,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'execution_time': execution_time,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'y_test': y_test,
        'y_pred': y_pred
    }

    if results_dir:
        # Save classification report
        report_path = os.path.join(results_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(classification_report(y_test, y_pred))
        print(f"Classification report saved to {report_path}")

        # Confusion Matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
                    yticklabels=['Class 0', 'Class 1'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        cm_plot_path = os.path.join(results_dir, 'confusion_matrix.png')
        plt.savefig(cm_plot_path)
        plt.close()
        print(f"Confusion matrix saved to {cm_plot_path}")

        # ROC-AUC curve plot (only for binary classification)
        if roc_auc is not None:
            fpr, tpr, thresholds = roc_curve(y_test, rf_classifier.predict_proba(X_test)[:, 1])
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.title('ROC Curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc='lower right')
            roc_auc_plot_path = os.path.join(results_dir, 'roc_auc_curve.png')
            plt.savefig(roc_auc_plot_path)
            plt.close()
            print(f"ROC-AUC curve saved to {roc_auc_plot_path}")

        # Plot feature importances
        feature_importance = rf_classifier.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        plt.figure(figsize=(8, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
        plt.title('Feature Importance')
        importance_plot_path = os.path.join(results_dir, 'feature_importance.png')
        plt.savefig(importance_plot_path)
        plt.close()
        print(f"Feature importance plot saved to {importance_plot_path}")

    # Memory usage information
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage = memory_info.rss / (1024 ** 2)  # in MB
    print(f"Memory Usage: {memory_usage:.2f} MB")

    # Save model and results as a dictionary
    model_path = os.path.join(results_dir, 'random_forest_model.pkl')
    joblib.dump(model_results, model_path)
    print(f"Random Forest model and results saved to {model_path}")

    return model_results


# Only run this if the script is executed directly (not imported)
if __name__ == "__main__":
    # Load preprocessed data
    preprocessed_data_path = "E:\\web-data-mining\\results\\preprocessed_data_concise.csv"
    prior_data = pd.read_csv(preprocessed_data_path)
    print(prior_data.columns)  # Print the column names to help identify the target column
    results_dir = "E:\\web-data-mining\\results"
    os.makedirs(results_dir, exist_ok=True)

    # Assuming you want to predict the 'target' column (replace 'target' with your actual target column name)
    if len(prior_data) >= 5:
        target_column = 'reordered'  # Updated to 'reordered' or the appropriate target column
        model_results = random_forest_classification(
            prior_data, target_column, results_dir=results_dir)

    else:
        print(f"Not enough samples to run classification. Only {len(prior_data)} rows available.")
