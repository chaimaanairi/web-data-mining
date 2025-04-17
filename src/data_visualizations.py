import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sns.set(style="whitegrid")


def orders_by_day(orders):
    plt.figure(figsize=(8, 4))
    sns.countplot(x='order_dow', data=orders, palette="Blues")
    plt.title("Orders by Day of Week")
    plt.xlabel("Day of Week (0=Sunday, 6=Saturday)")
    plt.ylabel("Order Count")
    st.pyplot(plt.gcf())


def orders_by_hour(orders):
    if 'order_hour_of_day' not in orders.columns:
        st.error("⚠️ 'order_hour_of_day' column not found in orders dataset.")
        st.write("Columns available:", orders.columns.tolist())
        return

    plt.figure(figsize=(10, 4))
    sns.countplot(x='order_hour_of_day', data=orders, palette="coolwarm")
    plt.title("Orders by Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Order Count")
    plt.xticks(rotation=90)
    st.pyplot(plt.gcf())


def top_10_most_ordered(order_products_prior, products):
    top_products = order_products_prior['product_id'].value_counts().head(10).reset_index()
    top_products.columns = ['product_id', 'count']
    top_products = top_products.merge(products, on='product_id')

    plt.figure(figsize=(10, 6))
    sns.barplot(x='count', y='product_name', data=top_products, palette="viridis")
    plt.title("Top 10 Most Ordered Products")
    plt.xlabel("Number of Orders")
    plt.ylabel("Product Name")
    st.pyplot(plt.gcf())


def avg_days_between_orders(orders):
    user_days = orders.groupby("user_id")["days_since_prior_order"].mean().reset_index()
    plt.figure(figsize=(8, 5))
    sns.histplot(user_days["days_since_prior_order"], bins=30, kde=True, color='skyblue')
    plt.title("Average Days Between Orders per User")
    plt.xlabel("Days")
    plt.ylabel("Number of Users")
    st.pyplot(plt.gcf())


def reorder_frequency(order_products_prior):
    plt.figure(figsize=(6, 4))
    order_products_prior["reordered"].value_counts().plot(kind="bar", color=["skyblue", "salmon"])
    plt.title("Reorder Frequency")
    plt.xticks([0, 1], ["Not Reordered", "Reordered"], rotation=0)
    plt.ylabel("Count")
    st.pyplot(plt.gcf())


def most_reordered_products(order_products_prior, products):
    reordered_prods = order_products_prior.groupby("product_id")["reordered"].mean().reset_index()
    reordered_prods = reordered_prods.merge(products, on="product_id")
    top_reordered = reordered_prods.sort_values("reordered", ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="reordered", y="product_name", data=top_reordered, palette="crest")
    plt.title("Top 10 Most Reordered Products")
    plt.xlabel("Reorder Ratio")
    plt.ylabel("Product Name")
    st.pyplot(plt.gcf())


def top_departments(order_products_prior, products, departments):
    prod_dept = products.merge(departments, on='department_id')
    dept_counts = order_products_prior.merge(prod_dept, on='product_id') \
                                      .groupby('department')['product_id'].count() \
                                      .sort_values(ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    dept_counts.plot(kind='barh', color='plum')
    plt.title("Top 10 Departments by Product Orders")
    plt.xlabel("Number of Products Ordered")
    plt.gca().invert_yaxis()
    st.pyplot(plt.gcf())


def top_aisles(order_products_prior, products, aisles, departments):
    prod_full = products.merge(aisles, on="aisle_id").merge(departments, on="department_id")
    top_aisles = order_products_prior.merge(prod_full, on="product_id") \
                                     .groupby("aisle")["product_id"].count() \
                                     .sort_values(ascending=False).head(10)

    plt.figure(figsize=(10, 5))
    top_aisles.plot(kind="bar", color="cornflowerblue")
    plt.title("Top 10 Most Common Aisles")
    plt.ylabel("Number of Products Ordered")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(plt.gcf())

