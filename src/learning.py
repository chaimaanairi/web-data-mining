import os
import time
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import psutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


def run_model(model, model_name, X_train, X_test, y_train, y_test, results_dir, features):
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    end_time = time.time()

    execution_time = end_time - start_time
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    roc_auc = None
    if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f"[{model_name}] Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    os.makedirs(results_dir, exist_ok=True)

    # Save classification report
    report_txt = os.path.join(results_dir, f'{model_name}_report.txt')
    with open(report_txt, 'w') as f:
        f.write(classification_report(y_test, y_pred))

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{model_name}_confusion_matrix.png'))
    plt.close()

    # ROC Curve
    if roc_auc is not None:
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'{model_name} - ROC Curve')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{model_name}_roc_auc_curve.png'))
        plt.close()

    # Feature importance (XGBoost only)
    if model_name == "XGBoost" and hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(8, 6))
        sns.barplot(data=importance_df, x="Importance", y="Feature")
        plt.title("Feature Importance (XGBoost)")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{model_name}_feature_importance.png'))
        plt.close()

    # Memory usage
    mem_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    print(f"[{model_name}] Memory usage: {mem_usage:.2f} MB")

    # Save model + metadata
    joblib.dump({
        'model': model,
        'y_test': y_test,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'execution_time': execution_time,
        'roc_auc': roc_auc
    }, os.path.join(results_dir, f'{model_name}_model.pkl'))


def prepare_and_run(data, target_column, features, results_dir):
    X = data[features]
    y = data[target_column]

    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    run_model(knn, "KNN", X_train, X_test, y_train, y_test, results_dir, features)

    # XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    run_model(xgb, "XGBoost", X_train, X_test, y_train, y_test, results_dir, features)


# Run directly
if __name__ == "__main__":
    data_path = "E:\\web-data-mining\\results\\preprocessed_data_concise.csv"
    results_dir = "E:\\web-data-mining\\results"

    data = pd.read_csv(data_path)
    print(data.columns)

    if len(data) >= 5:
        features = ['total_orders', 'avg_days_between_orders', 'reorder_ratio', 'avg_basket_size']
        prepare_and_run(data, target_column='reordered', features=features, results_dir=results_dir)
    else:
        print("Not enough data.")
