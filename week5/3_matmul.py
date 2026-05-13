"""
EXERCISE 3 [CORE] — Matrix Multiplication
==========================================
Matrix multiplication is the engine of everything in deep learning.
Every linear layer, every attention score, every projection = matmul.

The rule: (m, n) @ (n, p) → (m, p)
The inner dimensions MUST match. Outer dimensions give you the output shape.

Fill in each TODO, then run: python 3_matmul.py
"""

import torch

print("── EXERCISE 3: Matrix Multiplication ────────────────────")

# ── Part 3a: Basic linear layer ──────────────────────────────────────────────
# A linear layer is just: output = input @ weight
# Input has shape  [samples, features_in]
# Weight has shape [features_in, features_out]
# Output has shape [samples, features_out]

X = torch.randn(5, 4)   # 5 samples, 4 input features each
W = torch.randn(4, 3)   # weight matrix: maps 4 → 3 features

# TODO 3a: Multiply X by W using the @ operator
result = X @ W  # YOUR CODE HERE

assert result is not None, "Fill in 3a!"
print(f"3a. X shape: {X.shape}, W shape: {W.shape}")
print(f"    result shape: {result.shape}")   # should be [5, 3]
assert result.shape == torch.Size([5, 3]), f"Expected [5,3], got {result.shape}"

# ── Part 3b: Attention scores ─────────────────────────────────────────────────
# The actual formula from the transformer paper:
#   scores = Q @ K.T
# Q and K both have shape [seq_len, d_k]
# K.T (K transposed) has shape [d_k, seq_len]
# Result: [seq_len, seq_len] — every token vs every other token

seq_len = 6
d_k     = 64

Q = torch.randn(seq_len, d_k)   # 6 query vectors, each 64-dimensional
K = torch.randn(seq_len, d_k)   # 6 key vectors, each 64-dimensional

# TODO 3b: Compute attention scores = Q @ K transposed
# Hint: K.T transposes K from [6,64] to [64,6]
scores = Q @ (K.T)  # YOUR CODE HERE

assert scores is not None, "Fill in 3b!"
print(f"\n3b. Q shape: {Q.shape}, K.T shape: {K.T.shape}")
print(f"    scores shape: {scores.shape}")   # should be [6, 6]
assert scores.shape == torch.Size([6, 6]), f"Expected [6,6], got {scores.shape}"

# ── Part 3c: Understand the error, then fix it ───────────────────────────────
# Two matrices of the same shape CANNOT be multiplied directly.
# (3,4) @ (3,4) fails because inner dims are 4 and 3 — they don't match.
# You need to transpose one of them.

A = torch.randn(3, 4)
B = torch.randn(3, 4)

try:
    broken = A @ B
except RuntimeError as e:
    print(f"\n3c. Expected error: {e}")

# TODO 3c: Fix the multiplication using a transpose.
# Two valid options:
#   A @ B.T  → (3,4) @ (4,3) → shape [3, 3]
#   A.T @ B  → (4,3) @ (3,4) → shape [4, 4]
# Pick ONE and explain in a comment why that shape makes sense.
fixed = A.T @ B  # YOUR CODE HERE

assert fixed is not None, "Fill in 3c!"
print(f"    Fixed result shape: {fixed.shape}")
print("\n✅ Exercise 3 passed!")
