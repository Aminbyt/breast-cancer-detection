from tkinter import N
import pandas as pd 
import os

def load_data(path=None):
    if path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(project_root, "data", "Breast_cancer_dataset.csv")
    else:
        csv_path = path

    csv_path = os.path.abspath(csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found at: {csv_path}\nCurrent working dir: {os.getcwd()}"
        )

    return pd.read_csv(csv_path)
