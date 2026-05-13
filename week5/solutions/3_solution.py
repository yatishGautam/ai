"""SOLUTION — Exercise 3: Matrix Multiplication"""
import torch

X = torch.randn(5, 4)
W = torch.randn(4, 3)

# @ is the matrix multiply operator in Python/PyTorch
# Rule: (m, n) @ (n, p) → (m, p)
# Here: (5, 4) @ (4, 3) → (5, 3)
# Inner dims match (both 4). Output shape = outer dims.
result = X @ W
print(f"3a. result shape: {result.shape}")   # [5, 3]

seq_len, d_k = 6, 64
Q = torch.randn(seq_len, d_k)
K = torch.randn(seq_len, d_k)

# K.T transposes K from [6, 64] → [64, 6]
# Q @ K.T: (6, 64) @ (64, 6) → (6, 6)
# Entry [i, j] = dot product of query i with key j
# = how much token i should attend to token j
scores = Q @ K.T
print(f"3b. scores shape: {scores.shape}")   # [6, 6]

A = torch.randn(3, 4)
B = torch.randn(3, 4)

# A @ B.T: (3,4) @ (4,3) → (3,3)
# This works because transposing B flips it to (4,3),
# so inner dimensions match: A's last dim (4) == B.T's first dim (4)
fixed = A @ B.T
print(f"3c. fixed shape: {fixed.shape}")     # [3, 3]
print("✅ Solution 3 correct!")
