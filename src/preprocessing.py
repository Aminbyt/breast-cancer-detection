import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_data (data):
    """
    _  Drop unnecessary columns and perform minimal cleaning.
    - Drops 'Unnamed: 32' if present.
    - Returns a new DataFrame (does not modify in-place).
    """
    data = data.copy()
    if "Unnamed: 32" in data.columns:
        data = data.drop(columns ="Unnamed: 32")
    return data


def encode_target (data , target_col = "diagnosis"):
    """
    Encode the diagnosis column into a binary column 'diagnosis_M' (1 = M, 0 = B).
    Keeps other columns unchanged.
    Returns a new DataFrame with the added 'diagnosis_M' and original `target_col` dropped.
    """
    data = data.copy()
    if target_col not in data.columns:
        raise ValueError (f"column '{target_col}' not found in dataframe")
    
    data["diagnosis_M"] = (data[target_col]=="M").astype(int)
    data = data.drop(columns=[target_col])

    return data


def split_data(data , target_col = "diagnosis_M",test_size = 0.2 ,random_state=42, drop_id=True,stratify=True):
    data = data()
    if drop_id and "id" in data.columns:
        data = data.drop(columns=["id"])

    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    
    x = data.drop(columns=[target_col])
    y = data[target_col]

    strat = y if stratify else None

    x_train , x_test , y_train , y_test = train_test_split(
        x.values,y.values , test_size=test_size , random_state=random_state,stratify=strat
    )
    return x_train , x_test , y_train , y_test

def scale_features(x_train , x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled =  scaler.transform(x_test)
    return x_test_scaled , x_train_scaled , scaler

