"""
EXERCISE 1 [WARM UP] — Creating Tensors
========================================
Five ways to create tensors. Each one has a real use in deep learning.
Fill in each TODO, then run: python 1_creating_tensors.py
"""

import torch

print("── EXERCISE 1: Creating Tensors ──────────────────────────")

# TODO 1a: Create a 4x4 matrix of zeros
# Used for: initializing empty output buffers, masking
# Hint: torch.zeros(rows, cols)
zeros_matrix = torch.zeros(4,4)  # YOUR CODE HERE

# TODO 1b: Create a 3x3 matrix of ones
# Used for: creating all-ones masks, testing
# Hint: torch.ones(rows, cols)
ones_matrix = torch.ones(3,3)  # YOUR CODE HERE

# TODO 1c: Create a 5x5 matrix of random values (normal distribution)
# Used for: weight initialization — every model starts with random weights!
# Hint: torch.randn(rows, cols)
random_matrix = torch.randn(5,5)  # YOUR CODE HERE

# TODO 1d: Create a sequence [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Used for: position indices in a transformer (positional encoding)
# Hint: torch.arange(start, end)  — end is exclusive
sequence = torch.arange(0,10)  # YOUR CODE HERE

# TODO 1e: Create 6 evenly spaced values between 0.0 and 1.0
# Used for: creating linearly spaced ranges, testing softmax outputs
# Hint: torch.linspace(start, end, num_points)
evenly_spaced = torch.linspace(0,1,6)  # YOUR CODE HERE

# ── Verification ───────────────────────────────────────────────────────────────
assert zeros_matrix  is not None, "Fill in 1a!"
assert ones_matrix   is not None, "Fill in 1b!"
assert random_matrix is not None, "Fill in 1c!"
assert sequence      is not None, "Fill in 1d!"
assert evenly_spaced is not None, "Fill in 1e!"

print(f"1a. zeros_matrix shape:   {zeros_matrix.shape}")    # torch.Size([4, 4])
print(f"1b. ones_matrix shape:    {ones_matrix.shape}")     # torch.Size([3, 3])
print(f"1c. random_matrix shape:  {random_matrix.shape}")   # torch.Size([5, 5])
print(f"1d. sequence:             {sequence}")              # tensor([0,1,...,9])
print(f"1e. evenly_spaced:        {evenly_spaced}")         # 6 values 0.0→1.0

assert zeros_matrix.shape  == torch.Size([4, 4]),  "1a: wrong shape, expected [4,4]"
assert ones_matrix.shape   == torch.Size([3, 3]),  "1b: wrong shape, expected [3,3]"
assert random_matrix.shape == torch.Size([5, 5]),  "1c: wrong shape, expected [5,5]"
assert len(sequence) == 10,                         "1d: sequence should have 10 elements"
assert len(evenly_spaced) == 6,                     "1e: should have 6 values"

print("\n✅ Exercise 1 passed!")
