"""
EXERCISE 6 [CORE] — Broadcasting
==================================
Broadcasting = PyTorch automatically expands smaller tensors to match larger ones.
You don't need to manually tile or repeat — it just works.

Examples of where this appears in practice:
  - Adding a bias vector to an entire batch: [32, 10] + [10] → [32, 10]
  - Scaling attention scores:                [8, 12, 128, 128] / scalar → same shape
  - Layer normalization mean subtraction

Fill in each TODO, then run: python 6_broadcasting.py
"""

import torch
import math

print("── EXERCISE 6: Broadcasting ──────────────────────────────")

# ── Part 6a: Add a bias vector to a batch of outputs ─────────────────────────
# After nn.Linear, we add a bias to every row.
# batch_output is [32, 10] but bias is [10].
# PyTorch broadcasts bias across all 32 rows automatically.

batch_output = torch.zeros(32, 10)
bias = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5,
                     0.6, 0.7, 0.8, 0.9, 1.0])   # shape [10]

# TODO 6a: Add bias to every row in batch_output
# The [10] bias broadcasts across [32, 10] → result is [32, 10]
result = None  # YOUR CODE HERE

assert result is not None, "Fill in 6a!"
print(f"6a. batch_output: {batch_output.shape}")
print(f"    bias:          {bias.shape}")
print(f"    result:        {result.shape}")         # should be [32, 10]
print(f"    First row: {result[0]}")                # should match bias values exactly
assert result.shape == torch.Size([32, 10])
assert torch.allclose(result[0], bias), "Each row should equal the bias vector"

# ── Part 6b: Scale attention scores ──────────────────────────────────────────
# In the transformer, we scale attention scores by 1/sqrt(d_k).
# This prevents large d_k values from making scores so large that
# softmax saturates (becomes near 0/1 everywhere), which kills gradients.
#
# The scalar just divides every element — broadcasting at its simplest.

d_k = 64
attention_scores = torch.randn(8, 12, 128, 128)   # [batch, heads, seq, seq]

# TODO 6b: Divide all scores by sqrt(d_k)
# Hint: attention_scores / math.sqrt(d_k)
scaled_scores = None  # YOUR CODE HERE

assert scaled_scores is not None, "Fill in 6b!"
print(f"\n6b. Before scaling — std: {attention_scores.std():.3f}")
print(f"    After scaling  — std: {scaled_scores.std():.3f}")
print(f"    Values are smaller → softmax won't saturate → better gradients!")
assert scaled_scores.shape == attention_scores.shape
# std should be reduced by factor of sqrt(64) = 8
assert scaled_scores.std() < attention_scores.std(), "Scaled scores should have smaller std"

print("\n✅ Exercise 6 passed!")
