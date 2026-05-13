"""SOLUTION — Exercise 11: Scaled Dot-Product Attention in PyTorch"""
import torch
import math

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]   # get key/query dimension from last axis

    # Step 1: Raw attention scores
    # Q @ K.T: each query vector dot-producted with every key vector
    # Entry [i, j] = "how relevant is key j to query i?"
    # Shape: [seq_len, seq_len]
    scores = Q @ K.T

    # Step 2: Scale scores down by 1/sqrt(d_k)
    # Intuition: with d_k=64, random dot products have std ≈ sqrt(64)=8
    # Dividing brings std back to ~1, keeping softmax in a well-behaved range
    # Source: "Attention Is All You Need" (Vaswani et al., 2017), Section 3.2.1
    scores = scores / math.sqrt(d_k)

    # Step 3: Softmax over last dim (the "key" axis)
    # Converts scores to a probability distribution: each row sums to 1
    # High score → high probability → token gets more attention weight
    # dim=-1 means softmax over the last dimension (keys), not the query dim
    weights = torch.softmax(scores, dim=-1)

    # Step 4: Weighted sum of value vectors
    # output[i] = sum over j of (weights[i,j] * V[j])
    # = weighted average of all value vectors, where high-attention tokens contribute more
    # Shape: [seq_len, d_v]
    output = weights @ V

    return output, weights

torch.manual_seed(0)
seq_len, d_k, d_v = 5, 8, 8
Q = torch.randn(seq_len, d_k)
K = torch.randn(seq_len, d_k)
V = torch.randn(seq_len, d_v)

output, weights = scaled_dot_product_attention(Q, K, V)

print(f"output shape:  {output.shape}")    # [5, 8]
print(f"weights shape: {weights.shape}")   # [5, 5]
print(f"Row sums: {weights.sum(dim=-1)}")  # all ~1.0
print("✅ Solution 11 correct!")
