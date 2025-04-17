import matplotlib.pyplot as plt
import seaborn as sns

# Function for Total Orders per User
def plot_total_orders(df):
    total_orders = df.groupby('user_id')['order_id'].count().sort_values(ascending=False)
    total_orders.plot(kind='bar', figsize=(10, 6), color='skyblue')
    plt.title('Total Orders per User')
    plt.xlabel('User ID')
    plt.ylabel('Total Orders')
    plt.show()

# Function for Average Days Between Orders per User
def plot_avg_days_between_orders(df):
    avg_days = df.groupby('user_id')['days_since_prior_order'].mean()
    avg_days.plot(kind='hist', bins=30, figsize=(10, 6), color='orange', edgecolor='black')
    plt.title('Average Days Between Orders per User')
    plt.xlabel('Average Days')
    plt.ylabel('Frequency')
    plt.show()

# Function for User Reorder Ratio Distribution
def plot_reorder_ratio(df):
    reorder_ratio = df.groupby('user_id')['reordered'].mean()
    reorder_ratio.plot(kind='hist', bins=30, figsize=(10, 6), color='green', edgecolor='black')
    plt.title('User Reorder Ratio Distribution')
    plt.xlabel('Reorder Ratio')
    plt.ylabel('Frequency')
    plt.show()

# Function for Average Basket Size per User
def plot_avg_basket_size(df):
    avg_basket_size = df.groupby('user_id')['order_id'].count()
    avg_basket_size.plot(kind='hist', bins=30, figsize=(10, 6), color='purple', edgecolor='black')
    plt.title('Average Basket Size per User')
    plt.xlabel('Basket Size')
    plt.ylabel('Frequency')
    plt.show()

# Function for Relationship Between Basket Size and Reorder Ratio
def plot_basket_size_vs_reorder_ratio(df):
    basket_size = df.groupby('user_id')['order_id'].count()
    reorder_ratio = df.groupby('user_id')['reordered'].mean()
    plt.figure(figsize=(10, 6))
    plt.scatter(basket_size, reorder_ratio, alpha=0.5, color='red')
    plt.title('Basket Size vs Reorder Ratio')
    plt.xlabel('Basket Size')
    plt.ylabel('Reorder Ratio')
    plt.show()

# Function for Correlation Heatmap of Engineered Features
def plot_correlation_heatmap(df):
    corr_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Engineered Features')
    plt.show()
