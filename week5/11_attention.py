"""
EXERCISE 11 [CHALLENGE] — Scaled Dot-Product Attention in PyTorch
==================================================================
This is the formula that powers every transformer ever built:

  Attention(Q, K, V) = softmax( Q @ K.T / sqrt(d_k) ) @ V

You already implemented this in NumPy. Now do it in PyTorch.
The difference: this version works with batches, supports autograd,
and can run on a GPU with zero changes.

Each step has a clear mathematical purpose — the comments explain WHY.

Fill in each TODO, then run: python 11_attention.py
"""

import torch
import math

print("── EXERCISE 11: Attention in PyTorch ─────────────────────")

def scaled_dot_product_attention(Q, K, V):
    """
    Inputs:
      Q: [seq_len, d_k]  — query vectors (what each token is looking for)
      K: [seq_len, d_k]  — key vectors   (what each token advertises)
      V: [seq_len, d_v]  — value vectors (the actual content to mix)

    Returns:
      output:  [seq_len, d_v]       — weighted mix of values
      weights: [seq_len, seq_len]   — attention distribution (rows sum to 1)
    """
    d_k = Q.shape[-1]

    # TODO Step 1: Compute raw attention scores
    # For each query, compute a dot product with every key.
    # Result shape: [seq_len, seq_len]
    # High score = "this query is similar to this key = pay attention here"
    # Hint: Q @ K.T
    scores = None  # YOUR CODE HERE

    # TODO Step 2: Scale by 1/sqrt(d_k)
    # Without this, larger d_k causes dot products to grow in magnitude,
    # which pushes softmax into regions where gradients are near zero.
    # Dividing by sqrt(d_k) keeps variance of scores ~1 regardless of d_k.
    # Hint: scores / math.sqrt(d_k)
    scores = None  # YOUR CODE HERE

    # TODO Step 3: Softmax over last dimension (dim=-1)
    # Converts raw scores into a probability distribution over keys.
    # Each row sums to 1.0 — these are the attention weights.
    # Hint: torch.softmax(scores, dim=-1)
    weights = None  # YOUR CODE HERE

    # TODO Step 4: Weighted sum of values
    # Each token's output = weighted average of ALL value vectors.
    # Tokens with high attention weight contribute more to the output.
    # Result shape: [seq_len, d_v]
    # Hint: weights @ V
    output = None  # YOUR CODE HERE

    return output, weights

# ── Test ──────────────────────────────────────────────────────────────────────
torch.manual_seed(0)
seq_len, d_k, d_v = 5, 8, 8

Q = torch.randn(seq_len, d_k)
K = torch.randn(seq_len, d_k)
V = torch.randn(seq_len, d_v)

output, weights = scaled_dot_product_attention(Q, K, V)

assert output  is not None, "Fill in all 4 steps!"
assert weights is not None, "Fill in all 4 steps!"

print(f"output shape:  {output.shape}")    # should be [5, 8]
print(f"weights shape: {weights.shape}")   # should be [5, 5]

row_sums = weights.sum(dim=-1)
print(f"Row sums (should all be ~1.0): {row_sums}")

assert output.shape  == torch.Size([5, 8]),  f"output: expected [5,8], got {output.shape}"
assert weights.shape == torch.Size([5, 5]),  f"weights: expected [5,5], got {weights.shape}"
assert torch.allclose(row_sums, torch.ones(seq_len), atol=1e-5), \
    "Each row of weights must sum to 1.0 (softmax property)"

print()
print("You just implemented the core of every LLM:")
print("GPT-4, BERT, Llama, Claude — this exact function runs")
print("billions of times during training.")

print("\n✅ Exercise 11 passed!")
