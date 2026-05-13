"""SOLUTION — Exercise 4: Reshaping"""
import torch

batch_of_images = torch.randn(64, 28, 28)

# .view() reinterprets the same memory with a new shape
# Total elements must stay the same: 64*28*28 == 64*784 ✓
# -1 can be used as a wildcard: .view(64, -1) also works
flattened = batch_of_images.view(64, 784)
print(f"4a. flattened: {flattened.shape}")   # [64, 784]

batch_size, seq_len, embed_dim, num_heads, head_dim = 8, 128, 768, 12, 64
x = torch.randn(batch_size, seq_len, embed_dim)

# Split the last dimension (768) into (12 heads × 64 dims/head)
# Total elements: 8*128*768 == 8*128*12*64 ✓
# Now each head has its own 64-dim slice of the embedding
x_heads = x.view(batch_size, seq_len, num_heads, head_dim)
print(f"4b. x_heads: {x_heads.shape}")   # [8, 128, 12, 64]

# .permute() reorders axes — like numpy.transpose but for N-D tensors
# Current axis order: (0=batch, 1=seq, 2=heads, 3=head_dim)
# We want:           (0=batch, 1=heads, 2=seq,  3=head_dim)
# So axis 2 (heads) moves to position 1, and axis 1 (seq) moves to position 2
# → .permute(0, 2, 1, 3)
# Why? Each head needs to see ALL tokens sequentially,
# so heads must be its own batch-like dimension.
x_ready = x_heads.permute(0, 2, 1, 3)
print(f"4c. x_ready: {x_ready.shape}")   # [8, 12, 128, 64]
print("✅ Solution 4 correct!")
