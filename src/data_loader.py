import pandas as pd
import os

def load_data(data_dir):
    # File paths
    aisles_path = os.path.join(data_dir, "aisles.csv")
    departments_path = os.path.join(data_dir, "departments.csv")
    products_path = os.path.join(data_dir, "products.csv")
    orders_path = os.path.join(data_dir, "orders.csv")
    order_products_prior_path = os.path.join(data_dir, "order_products__prior.csv")
    order_products_train_path = os.path.join(data_dir, "order_products__train.csv")

    # Load CSVs
    aisles = pd.read_csv(aisles_path)
    departments = pd.read_csv(departments_path)
    products = pd.read_csv(products_path)
    orders = pd.read_csv(orders_path)
    order_products_prior = pd.read_csv(order_products_prior_path)
    order_products_train = pd.read_csv(order_products_train_path)

    return {
        "aisles": aisles,
        "departments": departments,
        "products": products,
        "orders": orders,
        "order_products_prior": order_products_prior,
        "order_products_train": order_products_train
    }


data_dir = "../data"  # folder where data CSV files are
data = load_data(data_dir)

# Preview some data
print(data["orders"].head())
print(data["products"].head())