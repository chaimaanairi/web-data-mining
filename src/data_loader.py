import pandas as pd
import os


def load_data(data_dir):
    # Get the absolute path for the data directory
    data_dir = os.path.abspath(data_dir)

    # Define file paths
    aisles_path = os.path.join(data_dir, "aisles.csv")
    departments_path = os.path.join(data_dir, "departments.csv")
    products_path = os.path.join(data_dir, "products.csv")
    orders_path = os.path.join(data_dir, "orders.csv")
    order_products_prior_path = os.path.join(data_dir, "order_products__prior.csv")
    order_products_train_path = os.path.join(data_dir, "order_products__train.csv")

    # List of all paths to check
    file_paths = [
        aisles_path, departments_path, products_path,
        orders_path, order_products_prior_path, order_products_train_path
    ]

    # Check if all files exist, raise error if any are missing
    missing_files = [file for file in file_paths if not os.path.exists(file)]
    if missing_files:
        raise FileNotFoundError(f"Missing the following files: {', '.join(missing_files)}")

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


# Update data directory path to be absolute
data_dir = "E:\\web-data-mining\\data"  # Absolute path to the 'data' folder
try:
    data = load_data(data_dir)
    print("✅ Data loaded successfully!")

    # Preview some data
    print("\nOrders data:")
    print(data["orders"].head())

    print("\nProducts data:")
    print(data["products"].head())

except FileNotFoundError as e:
    print(f"❌ Error: {e}")
