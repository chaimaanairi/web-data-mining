import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd

# Function for Total Orders per User
def plot_total_orders(df):
    plt.figure(figsize=(10, 4))
    sns.histplot(df['total_orders'], bins=30, kde=True, color='teal')
    plt.title("Distribution of Total Orders per User")
    plt.xlabel("Total Orders")
    plt.ylabel("User Count")
    st.pyplot(plt.gcf())

# Function for Average Days Between Orders per User
def plot_avg_days_between_orders(df):
    plt.figure(figsize=(10, 4))
    sns.histplot(df['avg_days_between_orders'], bins=30, kde=True, color='steelblue')
    plt.title("Avg Days Between Orders (Users)")
    plt.xlabel("Avg Days Between Orders")
    plt.ylabel("User Count")
    st.pyplot(plt.gcf())

# Function for User Reorder Ratio Distribution
def plot_reorder_ratio(df):
    plt.figure(figsize=(10, 4))
    sns.histplot(df['reorder_ratio'], bins=30, kde=True, color='mediumseagreen')
    plt.title("User Reorder Ratio Distribution")
    plt.xlabel("Reorder Ratio")
    plt.ylabel("User Count")
    st.pyplot(plt.gcf())

# Function for Average Basket Size per User
def plot_avg_basket_size(df):
    plt.figure(figsize=(10, 4))
    sns.histplot(df['avg_basket_size'], bins=30, kde=True, color='orchid')
    plt.title("Average Basket Size per User")
    plt.xlabel("Avg Basket Size")
    plt.ylabel("User Count")
    st.pyplot(plt.gcf())

# Function for Relationship Between Basket Size and Reorder Ratio
def plot_basket_size_vs_reorder_ratio(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='avg_basket_size', y='reorder_ratio', alpha=0.4, color='coral')
    plt.title("Reorder Ratio vs. Avg Basket Size")
    plt.xlabel("Avg Basket Size")
    plt.ylabel("Reorder Ratio")
    st.pyplot(plt.gcf())

# Function for Correlation Heatmap of Engineered Features
def plot_correlation_heatmap(df):
    features = ['total_orders', 'avg_days_between_orders', 'reorder_ratio', 'avg_basket_size']

    numeric_features = [col for col in features if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    if not numeric_features:
        st.warning("No valid numeric features available for correlation.")
        return

    corr = df[numeric_features].corr()

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', square=True)
    plt.title("Correlation Between User-Level Features")
    st.pyplot(plt.gcf())
