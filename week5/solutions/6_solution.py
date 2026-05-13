"""SOLUTION — Exercise 6: Broadcasting"""
import torch
import math

batch_output = torch.zeros(32, 10)
bias = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5,
                     0.6, 0.7, 0.8, 0.9, 1.0])

# Broadcasting rule: PyTorch aligns shapes from the RIGHT
# batch_output: [32, 10]
# bias:               [10]   ← automatically expanded to [32, 10]
# Result: bias is added to every row independently — no explicit loop needed
result = batch_output + bias

print(f"6a. result shape: {result.shape}")     # [32, 10]
print(f"    First row: {result[0]}")           # matches bias exactly

d_k = 64
attention_scores = torch.randn(8, 12, 128, 128)

# Dividing by a scalar broadcasts trivially — every element gets divided
# This is the scaling step from "Attention Is All You Need":
# scores /= sqrt(d_k)
# Without it, as d_k grows, dot products grow proportionally,
# which saturates softmax and creates near-zero gradients.
scaled_scores = attention_scores / math.sqrt(d_k)

print(f"\n6b. Before std: {attention_scores.std():.3f}")
print(f"    After  std: {scaled_scores.std():.3f}")
print("✅ Solution 6 correct!")
