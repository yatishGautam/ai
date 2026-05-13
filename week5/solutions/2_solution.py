"""SOLUTION — Exercise 2: Shape Is Everything"""
import torch

batch_size = 8
seq_len    = 128
embed_dim  = 768
num_heads  = 12
head_dim   = embed_dim // num_heads   # 64

t1 = torch.randn(batch_size, seq_len, embed_dim)

# .numel() multiplies all dimensions together
# Equivalent to: 8 * 128 * 768 = 786,432
total_numbers = t1.numel()
# Alternative: total_numbers = 8 * 128 * 768

print(f"Total numbers in t1: {total_numbers}")   # 786432
print("✅ Solution 2 correct!")
