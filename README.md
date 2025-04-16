# Customer Segmentation and Behavior Analysis Based on Web Usage Data

## Overview
This project analyzes user behavior data from web-based shopping platforms (e.g., InstaCart) to perform customer segmentation and predict purchasing trends. By categorizing users based on their shopping habits, personalized marketing strategies and efficient recommendation systems can be developed. 

The dataset used in this project contains anonymized data about users, products, and their order histories. Several machine learning algorithms are applied to segment customers, predict purchasing tendencies, and analyze relationships between products.

## Table of Contents
- [Project Objective](#project-objective)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Algorithms Used](#algorithms-used)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Analysis and Visualization](#analysis-and-visualization)
- [GUI (Streamlit)](#gui-streamlit)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

---

## Project Objective
The goal of this project is to:
- **Analyze user behavior** using web-based shopping data.
- **Segment customers** based on their shopping habits and predict their purchasing behaviors.
- **Develop personalized marketing strategies** and improve recommendation systems for e-commerce platforms.

---

## Dataset
The **InstaCart Online Grocery Basket Analysis Dataset** will be used for this project. It contains data about previous orders, product categories, and user interactions on an e-commerce platform.

### Dataset Files:
- `aisles.csv`: Information about aisles where products belong.
- `departments.csv`: Information about the departments to which products belong.
- `order_products__prior.csv`: Products purchased in prior orders.
- `order_products__train.csv`: Products in the training set for model training.
- `orders.csv`: Order information (e.g., timestamps, user data).
- `products.csv`: Information about the products (e.g., name, price, category).

### Data Types:
- Structured data (numerical, categorical)
- Customer interaction data
- Purchase history

---

## Methodology

### Data Preprocessing
Before applying machine learning algorithms, we perform essential preprocessing steps:
- **Handle missing values** (e.g., using mean filling or SMOTE for imbalanced data).
- **Convert categorical data** to numerical format using **One-Hot Encoding** or **Label Encoding**.
- **Time-based analysis** to identify trends such as purchase frequency over time.

### Algorithms Used
- **K-Means Clustering:** For customer segmentation.
- **K-Nearest Neighbors (KNN):** For instance-based classification.
- **Random Forest and XGBoost:** Tree-based algorithms for classification tasks (e.g., predicting product purchases).
- **DBSCAN:** Density-based clustering for identifying patterns in data with varying densities.
- **Apriori Algorithm:** Association rule mining for uncovering frequent product pairs.

### Evaluation Metrics
- **Classification:** Accuracy, Precision, Recall, F1-Score.
- **Clustering:** Silhouette Score.
- **Additional Metrics:** Confusion Matrix, ROC-AUC, Execution Time, Memory Usage.

### Analysis and Visualization
- **Customer Segments:** 2D clustering plots.
- **Predictions:** Confusion matrices, ROC curves.
- **Association Rules:** Heatmaps or tables.
- **Performance Comparison:** Visual comparisons of runtime and memory usage for each algorithm.

---

## GUI (Streamlit)
A user-friendly, web-based **Streamlit** interface will allow users to:
- Upload datasets in CSV format.
- Select and configure algorithms for clustering, classification, and association rule mining.
- View results in graphical or tabular form.
- Compare the performance of different algorithms based on metrics like accuracy, runtime, and memory usage.

---

## Installation

To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/customer-segmentation-web-usage.git
   cd customer-segmentation-web-usage
   
2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
    streamlit run app/main.py
    ```

4. **Using the GUI:**
- Open the application in your web browser.
- Upload your dataset in CSV format.
- Select the algorithm(s) you want to run.
- Configure the parameters for each algorithm.
- View the results in graphical or tabular form.
- Compare algorithm performance metrics.

## Usage

1. **Running the Streamlit GUI:**
- Open your web browser and navigate to `http://localhost:8501` to access the Streamlit app

2. **Using the GUI:**
- Open the application in your web browser.
- Upload your dataset in CSV format.
- Select the algorithm(s) you want to run.
- Configure the parameters for each algorithm.
- View the results in graphical or tabular form. 
- Compare algorithm performance metrics.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Note: 
- Make sure you have a working Python environment and have installed all the necessary libraries.
- If you face issues with missing data or need help with algorithm configuration, feel free to refer to the detailed documentation or contact the project maintainer.


### Key Changes:
- **Corrected formatting of the Installation section** by fixing the code block and ensuring it was consistent.
- **Expanded Usage instructions** to clarify the steps for using the GUI.

This README should now provide clear instructions for setting up the project, running it locally, and using the GUI. Feel free to modify any parts as needed!



