# src/visualizations.py

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def show_raw_data_viz(data_dict):
    orders = data_dict['orders']
    products = data_dict['products']

    st.subheader("Order Hour of Day Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(orders['order_hour_of_day'], bins=24, kde=False, ax=ax1)
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Order Count")
    st.pyplot(fig1)

    st.subheader("Days Since Prior Order Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(orders['days_since_prior_order'], bins=30, kde=False, ax=ax2)
    ax2.set_xlabel("Days Since Prior Order")
    ax2.set_ylabel("Count")
    st.pyplot(fig2)

    st.subheader("Top Products")
    top_products = products['product_name'].value_counts().head(10)
    fig3, ax3 = plt.subplots()
    sns.barplot(x=top_products.values, y=top_products.index, ax=ax3)
    ax3.set_xlabel("Frequency")
    ax3.set_ylabel("Product")
    st.pyplot(fig3)


def show_preprocessed_data_viz(df):
    st.subheader("Reorder Ratio Distribution")
    fig4, ax4 = plt.subplots()
    sns.histplot(df['reorder_ratio'], bins=20, kde=True, ax=ax4)
    ax4.set_xlabel("Reorder Ratio")
    st.pyplot(fig4)

    st.subheader("Average Basket Size")
    fig5, ax5 = plt.subplots()
    sns.histplot(df['avg_basket_size'], bins=20, kde=True, ax=ax5)
    ax5.set_xlabel("Avg Basket Size")
    st.pyplot(fig5)

    st.subheader("Total Orders per User")
    fig6, ax6 = plt.subplots()
    sns.histplot(df['total_orders'], bins=20, kde=False, ax=ax6)
    ax6.set_xlabel("Total Orders")
    st.pyplot(fig6)
