import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for better visuals
sns.set(style="whitegrid")


# Function to load the data
def load_data(data_path):
    """Load the preprocessed CSV data."""
    return pd.read_csv(data_path)


# Function to plot the distribution of Total Orders per User
def plot_total_orders(df):
    """Plot the distribution of Total Orders per User."""
    plt.figure(figsize=(10, 4))
    sns.histplot(df['total_orders'], bins=30, kde=True, color='teal')
    plt.title("Distribution of Total Orders per User")
    plt.xlabel("Total Orders")
    plt.ylabel("User Count")
    plt.show()


# Function to plot Average Days Between Orders per User
def plot_avg_days_between_orders(df):
    """Plot the distribution of Average Days Between Orders per User."""
    plt.figure(figsize=(10, 4))
    sns.histplot(df['avg_days_between_orders'], bins=30, kde=True, color='steelblue')
    plt.title("Avg Days Between Orders (Users)")
    plt.xlabel("Avg Days Between Orders")
    plt.ylabel("User Count")
    plt.show()


# Function to plot Reorder Ratio Distribution
def plot_reorder_ratio(df):
    """Plot the distribution of User Reorder Ratio."""
    plt.figure(figsize=(10, 4))
    sns.histplot(df['reorder_ratio'], bins=30, kde=True, color='mediumseagreen')
    plt.title("User Reorder Ratio Distribution")
    plt.xlabel("Reorder Ratio")
    plt.ylabel("User Count")
    plt.show()


# Function to plot Average Basket Size per User
def plot_avg_basket_size(df):
    """Plot the distribution of Average Basket Size per User."""
    plt.figure(figsize=(10, 4))
    sns.histplot(df['avg_basket_size'], bins=30, kde=True, color='orchid')
    plt.title("Average Basket Size per User")
    plt.xlabel("Avg Basket Size")
    plt.ylabel("User Count")
    plt.show()


# Function to plot the relationship between Basket Size and Reorder Ratio
def plot_basket_size_vs_reorder_ratio(df):
    """Plot the relationship between Basket Size and Reorder Ratio."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='avg_basket_size', y='reorder_ratio', alpha=0.4, color='coral')
    plt.title("Reorder Ratio vs. Avg Basket Size")
    plt.xlabel("Avg Basket Size")
    plt.ylabel("Reorder Ratio")
    plt.show()


# Function to plot the correlation heatmap of engineered features
def plot_correlation_heatmap(df):
    """Plot a heatmap showing correlations between engineered features."""
    features = ['total_orders', 'avg_days_between_orders', 'reorder_ratio', 'avg_basket_size']
    corr = df[features].corr()

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', square=True)
    plt.title("Correlation Between User-Level Features")
    plt.show()


