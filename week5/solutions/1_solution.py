"""SOLUTION — Exercise 1: Creating Tensors"""
import torch

# torch.zeros(rows, cols) → fills every element with 0.0
# Used for initializing empty buffers, masks, or accumulators
zeros_matrix = torch.zeros(4, 4)

# torch.ones(rows, cols) → fills every element with 1.0
# Used for all-ones masks or baseline comparisons
ones_matrix = torch.ones(3, 3)

# torch.randn(rows, cols) → random values from N(0,1) (mean=0, std=1)
# This is how model weights are initialized — small random starting values
random_matrix = torch.randn(5, 5)

# torch.arange(start, end) → integers from start up to (not including) end
# Used for position indices: tokens 0, 1, 2, ..., seq_len-1
sequence = torch.arange(0, 10)

# torch.linspace(start, end, n) → n evenly spaced floats including both endpoints
# Used for creating smooth ranges or testing softmax output distributions
evenly_spaced = torch.linspace(0, 1, 6)

print(f"1a. zeros_matrix shape:   {zeros_matrix.shape}")    # [4, 4]
print(f"1b. ones_matrix shape:    {ones_matrix.shape}")     # [3, 3]
print(f"1c. random_matrix shape:  {random_matrix.shape}")   # [5, 5]
print(f"1d. sequence:             {sequence}")              # 0..9
print(f"1e. evenly_spaced:        {evenly_spaced}")         # 0.0..1.0
print("✅ Solution 1 correct!")
