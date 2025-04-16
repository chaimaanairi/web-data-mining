from src.data_loader import load_data
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(data):
    # Merge products with aisles and departments
    products = data['products'].merge(data['aisles'], on='aisle_id', how='left')
    products = products.merge(data['departments'], on='department_id', how='left')

    # Encode categorical columns (Label Encoding)
    label_encoders = {}
    for col in ['product_name', 'aisle', 'department']:
        le = LabelEncoder()
        products[col] = le.fit_transform(products[col])
        label_encoders[col] = le  # store for inverse_transform if needed

    # Combine prior orders with products
    prior = data['order_products_prior'].merge(products, on='product_id', how='left')
    prior = prior.merge(data['orders'], on='order_id', how='left')

    # Fill missing values
    prior['days_since_prior_order'] = prior['days_since_prior_order'].fillna(0)

    # --- Feature Engineering ---

    # Total orders per user
    user_order_counts = data['orders'].groupby('user_id')['order_id'].count().reset_index()
    user_order_counts.columns = ['user_id', 'total_orders']

    # Average days between orders per user
    avg_days = data['orders'].groupby('user_id')['days_since_prior_order'].mean().reset_index()
    avg_days.columns = ['user_id', 'avg_days_between_orders']
    avg_days['avg_days_between_orders'] = avg_days['avg_days_between_orders'].fillna(0)

    # Merge user-level features into prior
    prior = prior.merge(user_order_counts, on='user_id', how='left')
    prior = prior.merge(avg_days, on='user_id', how='left')

    # Reorder ratio per user
    reorder_ratio = prior.groupby('user_id')['reordered'].mean().reset_index()
    reorder_ratio.columns = ['user_id', 'reorder_ratio']
    prior = prior.merge(reorder_ratio, on='user_id', how='left')

    # Average basket size per user
    basket_size = prior.groupby(['user_id', 'order_id'])['product_id'].count().groupby('user_id').mean().reset_index()
    basket_size.columns = ['user_id', 'avg_basket_size']
    prior = prior.merge(basket_size, on='user_id', how='left')

    return prior


data = load_data("../data")
prior_data = preprocess_data(data)


print(prior_data[['user_id', 'order_id', 'reordered', 'total_orders', 'avg_days_between_orders', 'reorder_ratio', 'avg_basket_size']].head())

