import numpy as np
import pandas as pd

# 1) Load CSV (change filename/path as needed)
df = pd.read_csv("house-prices.csv")  # e.g. "Housing.csv" or whatever your file is called
print(df.head())
print(df.columns)

# 2) Extract columns
X = df["SqFt"].to_numpy(dtype=np.float64)
y = df["Price"].to_numpy(dtype=np.float64)

# 3) Standardize X for stable training
X_mean, X_std = X.mean(), X.std()
Xn = (X - X_mean) / X_std

# 4) Single neuron model: y_hat = w*x + b
rng = np.random.default_rng(42)
w = rng.normal()
b = 0.0

def forward(x, w, b):
    return w * x + b

def mse(y_hat, y_true):
    return np.mean((y_hat - y_true) ** 2)

# 5) Train with gradient descent
lr = 0.05
steps = 2000
n = len(Xn)

for step in range(1, steps + 1):
    y_hat = forward(Xn, w, b)
    loss = mse(y_hat, y)

    # gradients
    dL_dyhat = (2.0 / n) * (y_hat - y)
    grad_w = np.sum(dL_dyhat * Xn)
    grad_b = np.sum(dL_dyhat)

    # update
    w -= lr * grad_w
    b -= lr * grad_b

    if step % 100 == 0:
        print(f"step {step:4d} | loss {loss:,.2f} | w {w:,.4f} | b {b:,.2f}")

# 6) Convert learned params back to original units ($/sqft)
w_dollars_per_sqft = w / X_std
b_base_price = b - (w * X_mean / X_std)

print("\nLearned relationship (original units):")
print(f"Estimated $/sqft: {w_dollars_per_sqft:,.2f}")
print(f"Estimated base price: {b_base_price:,.2f}")



