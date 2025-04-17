import tkinter as tk
from tkinter import messagebox
import pandas as pd
import os

DATA_DIR = "E:/web-data-mining/data"

class DataApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Customer Segmentation App")
        self.root.geometry("600x400")

        self.orders_df = None
        self.products_df = None

        self.label = tk.Label(root, text="Customer Segmentation", font=("Helvetica", 16))
        self.label.pack(pady=10)

        self.load_button = tk.Button(root, text="📂 Auto-Load Data from /data", command=self.load_data)
        self.load_button.pack(pady=10)

        self.preview_button = tk.Button(root, text="👁️ Preview Loaded Data", command=self.preview_data, state=tk.DISABLED)
        self.preview_button.pack(pady=10)

    def load_data(self):
        try:
            self.orders_df = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"))
            self.products_df = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
            messagebox.showinfo("Success", "✅ Data loaded successfully!")
            self.preview_button.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"❌ Failed to load data: {e}")

    def preview_data(self):
        if self.orders_df is not None and self.products_df is not None:
            preview_window = tk.Toplevel(self.root)
            preview_window.title("📋 Data Preview")
            text = tk.Text(preview_window, width=100, height=25)
            text.pack()
            text.insert(tk.END, "📦 Orders Data (first 5 rows):\n")
            text.insert(tk.END, self.orders_df.head().to_string())
            text.insert(tk.END, "\n\n🛒 Products Data (first 5 rows):\n")
            text.insert(tk.END, self.products_df.head().to_string())
        else:
            messagebox.showwarning("Warning", "⚠️ Load the data first!")

if __name__ == "__main__":
    root = tk.Tk()
    app = DataApp(root)
    root.mainloop()
