"""
EXERCISE 4 [CORE] — Reshaping: view() and permute()
====================================================
Reshaping is how one block of memory gets interpreted different ways.
It's central to multi-head attention: you split the embedding dim
into N heads, then rearrange axes so each head can process all tokens.

view()    — changes shape, same memory, dimensions must be compatible
permute() — reorders axes (like numpy transpose for N-D tensors)

Fill in each TODO, then run: python 4_reshaping.py
"""

import torch

print("── EXERCISE 4: Reshaping ─────────────────────────────────")

# ── Part 4a: Flatten images ───────────────────────────────────────────────────
# MNIST images are 28×28 pixels.
# A fully-connected layer expects a flat vector, not a 2D grid.
# So we reshape [64, 28, 28] → [64, 784]

batch_of_images = torch.randn(64, 28, 28)   # 64 images, each 28×28

# TODO 4a: Reshape to [64, 784] using .view()
# Hint: 28 * 28 = 784
# Hint: .view(batch_size, new_dim)
flattened = batch_of_images.view(64, 784)
print(f"flattened shape {flattened.shape}")

assert flattened is not None, "Fill in 4a!"
print(f"4a. Before: {batch_of_images.shape}")
print(f"    After:  {flattened.shape}")    # should be [64, 784]
assert flattened.shape == torch.Size([64, 784])

# view() does NOT copy data — it's a new view of the same memory
print(f"    Same memory? {batch_of_images.data_ptr() == flattened.data_ptr()}")  # True

# ── Part 4b: Split embedding into heads ──────────────────────────────────────
# This is the EXACT operation inside every multi-head attention implementation.
# We have a combined [batch, seq, embed] tensor.
# We want to split embed_dim into num_heads × head_dim.
# Result: [batch, seq, num_heads, head_dim]

batch_size = 8
seq_len    = 128
embed_dim  = 768
num_heads  = 12
head_dim   = 64    # 768 / 12 = 64

x = torch.randn(batch_size, seq_len, embed_dim)

# TODO 4b: Reshape x to [8, 128, 12, 64]
# Hint: x.view(batch_size, seq_len, num_heads, head_dim)
x_heads = x.view(batch_size, seq_len, num_heads, head_dim)

assert x_heads is not None, "Fill in 4b!"
print(f"\n4b. Before: {x.shape}")
print(f"    After:  {x_heads.shape}")    # should be [8, 128, 12, 64]
assert x_heads.shape == torch.Size([8, 128, 12, 64])

# ── Part 4c: Permute so heads come before tokens ──────────────────────────────
# We need [batch, num_heads, seq_len, head_dim] = [8, 12, 128, 64]
# because each attention head operates independently over ALL tokens.
# Current axis order: (0=batch, 1=seq, 2=heads, 3=head_dim)
# Desired axis order: (0=batch, 1=heads, 2=seq,  3=head_dim)
# So we permute: 0 stays, 2 moves to position 1, 1 moves to position 2, 3 stays
# → .permute(0, 2, 1, 3)

# TODO 4c: Permute x_heads from [8, 128, 12, 64] to [8, 12, 128, 64]
# Hint: .permute(0, 2, 1, 3)
x_ready = x_heads.permute(0 , 2, 1, 3)  # YOUR CODE HERE

assert x_ready is not None, "Fill in 4c!"
print(f"\n4c. Before permute: {x_heads.shape}")
print(f"    After permute:  {x_ready.shape}")    # should be [8, 12, 128, 64]
assert x_ready.shape == torch.Size([8, 12, 128, 64])

print("\n    This is EXACTLY what happens inside every transformer!")
print("✅ Exercise 4 passed!")
