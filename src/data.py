# Dataset utilities
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_csv_dataset(path, label_column):
    df = pd.read_csv(path)
    y = df[label_column].values
    X = df.drop(columns=[label_column]).values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return train_test_split(X, y, test_size=0.2, random_state=42)
