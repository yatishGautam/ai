import numpy as np
import pandas as pd
import os

# Load the dataset
base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, 'house-prices.csv')

df = pd.read_csv(file_path)
print(df.head())
print(df.info())
print(df.columns)

# Extract columns
X = df["SqFt"].to_numpy(dtype=np.float64)
y = df["Price"].to_numpy(dtype=np.float64)

