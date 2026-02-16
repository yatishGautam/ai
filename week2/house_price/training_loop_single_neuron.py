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

X_mean, X_std = X.mean(), X.std();
Xn = (X - X_mean) / X_std

# Single neuron model: y_hat = w*x + b
rng = random.default_rng(42)
w = rng.normal()
b = 0.0

def forward(w, x , b):
    return w * x + b

def mse(y_hat, y_true):
    return np.mean((y_hat - y_true) ** 2)

#train with gradient disecent

lr = 0.05
steps = 2000
n = len(Xn)

for step in range(1, steps+1):
    y_hat = forward(w, Xn, b)
    loss = mse(y_hat, y)

    #gradient
    dL_dYhat = (2/n)*(y_hat-y)
    grad_w = np.sum(dL_dYhat * Xn)
    grad_b = np.sum(dL_dYhat)

    #update
    w = w - lr * grad_w
    b = b - lr * grad_b

    if step % 100 == 0:
        print(f"step {step:4d} | loss {loss:,.2f} | w {w:,.4f} | b {b:,.2f}")
# Convert learned params back to original units ($/sqft)
w_dollars_per_sqft = w / X_std
base_dir_price = b - (w * X_mean / X_std)


