"""
EXERCISE 2 [WARM UP] — Shape Is Everything
===========================================
Reading shapes is the most important skill for understanding model code.
Every tensor in a transformer has a specific meaning behind each dimension.
Fill in each TODO, then run: python 2_shapes.py
"""

import torch

print("── EXERCISE 2: Shape Is Everything ──────────────────────")

batch_size = 8
seq_len    = 128
embed_dim  = 768
num_heads  = 12
head_dim   = embed_dim // num_heads   # 768 / 12 = 64

t1 = torch.randn(batch_size, seq_len, embed_dim)
t2 = torch.randn(batch_size, seq_len, num_heads, head_dim)
t3 = torch.randn(batch_size, num_heads, seq_len, head_dim)
t4 = torch.randn(num_heads, seq_len, seq_len)

print(f"t1 shape: {t1.shape}")   # [8, 128, 768]
print(f"t2 shape: {t2.shape}")   # [8, 128, 12, 64]
print(f"t3 shape: {t3.shape}")   # [8, 12, 128, 64]
print(f"t4 shape: {t4.shape}")   # [12, 128, 128]

# ── Written answers (just read these, no code needed) ─────────────────────────
#
#  t1 = [8, 128, 768]
#         ^   ^    ^
#         |   |    └── embedding dimension (768 features per token)
#         |   └─────── sequence length (128 tokens per sequence)
#         └──────────── batch size (8 sequences processed in parallel)
#
#  t4 = [12, 128, 128]
#          ^    ^    ^
#          |    |    └── key position (which token is being attended TO)
#          |    └─────── query position (which token is doing the attending)
#          └──────────── attention head index
#
#  WHY is t4 128×128?
#  Because in self-attention, every token attends to every other token.
#  128 queries × 128 keys = 128×128 score matrix, one per head.

# TODO 2: How many total numbers are in t1?
# Two ways to do this:
#   Option A: multiply all dimensions:  8 * 128 * 768
#   Option B: use the built-in method:  t1.numel()
total_numbers = 786432  # YOUR CODE HERE

# ── Verification ───────────────────────────────────────────────────────────────
assert total_numbers is not None, "Fill in the TODO!"
print(f"\nTotal numbers in t1: {total_numbers}")   # should be 786432

assert total_numbers == 786432, f"Expected 786432, got {total_numbers}"
print("\n✅ Exercise 2 passed!")
