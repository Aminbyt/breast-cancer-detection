import pandas as pd 

def load_data(path=r"data/Breast_cancer_dataset.csv"):
    return pd.read_csv(path)
