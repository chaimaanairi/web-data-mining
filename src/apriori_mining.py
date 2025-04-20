import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

def run_apriori(data, results_dir,
                min_support=0.001,
                min_confidence=0.05,
                min_lift=1.5,
                min_product_count=100,
                sample_size=50000):
    # Filter infrequent products
    product_counts = data['product_name'].value_counts()
    frequent_products = product_counts[product_counts >= min_product_count].index
    data = data[data['product_name'].isin(frequent_products)]

    # Group products per order
    grouped = data.groupby('order_id')['product_name'].apply(list)

    # Sample orders to reduce memory usage
    if len(grouped) > sample_size:
        grouped = grouped.sample(n=sample_size, random_state=42)

    # Transaction Encoding
    te = TransactionEncoder()
    te_ary = te.fit(grouped).transform(grouped)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Apriori algorithm
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    print(f"Frequent itemsets (support >= {min_support}):")
    print(frequent_itemsets)

    # Association rules with min_confidence and min_lift
    rules = association_rules(frequent_itemsets,
                              metric="confidence",
                              min_threshold=min_confidence)
    # Filter rules based on lift
    rules = rules[rules['lift'] >= min_lift]

    # Save rules to CSV
    os.makedirs(results_dir, exist_ok=True)
    rules_path = os.path.join(results_dir, "apriori_rules.csv")
    rules.to_csv(rules_path, index=False)

    # Metrics
    num_rules = len(rules)
    avg_support = rules['support'].mean()
    avg_confidence = rules['confidence'].mean()
    avg_lift = rules['lift'].mean()
    lift_coverage = (rules['lift'] >= min_lift).mean() * 100

    # Save the metrics to a report text file
    apriori_report_path = os.path.join(results_dir, "apriori_report.txt")
    with open(apriori_report_path, 'w') as f:
        f.write("Apriori Algorithm Report\n")
        f.write("=========================\n")
        f.write(f"Number of Rules: {num_rules}\n")
        f.write(f"Average Support: {avg_support:.4f}\n")
        f.write(f"Average Confidence: {avg_confidence:.4f}\n")
        f.write(f"Average Lift: {avg_lift:.4f}\n")
        f.write(f"Lift Coverage (Lift >= 1.5): {lift_coverage:.2f}%\n")
        f.write("\nMetrics and results have been saved successfully.\n")
    print(f"✅ Apriori report saved to {apriori_report_path}")

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=rules, x='support', y='confidence', hue='lift', palette='viridis')
    plt.title('Support vs Confidence (Colored by Lift)')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.savefig(os.path.join(results_dir, "support_vs_confidence.png"))
    plt.close()

    # Lift Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(rules['lift'], bins=50, kde=True, color='skyblue')
    plt.title('Distribution of Lift')
    plt.xlabel('Lift')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(results_dir, "lift_histogram.png"))
    plt.close()

    print(f"✅ {num_rules} rules generated and saved to {rules_path}")

    # Return metrics and rules
    metrics = {
        'num_rules': num_rules,
        'avg_support': avg_support,
        'avg_confidence': avg_confidence,
        'avg_lift': avg_lift,
        'lift_coverage': lift_coverage
    }

    return rules, metrics

if __name__ == "__main__":
    data_path = "E:\\web-data-mining\\results\\preprocessed_data_concise.csv"
    results_dir = "E:\\web-data-mining\\results"

    data = pd.read_csv(data_path)
    if 'order_id' in data.columns and 'product_name' in data.columns:
        rules, metrics = run_apriori(data, results_dir)
        print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())
        print(metrics)
    else:
        print("❌ Required columns not found.")
