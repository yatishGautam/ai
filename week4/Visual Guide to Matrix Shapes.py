"""
Visual guide to understand matrix shapes through the attention computation.

Let's trace through with concrete numbers:
- seq_len = 5 (5 tokens: "Hello how are you ?")
- d_model = 16 (each token has 16-dimensional embedding)
- d_k = 4 (we project down to 4 dimensions)
"""

print("=== SHAPE TRANSFORMATIONS ===\n")

# Input
print("X shape: (5, 16)")
print("  ↓ 5 tokens, each with 16-dim embedding")
print()

# Projections
print("W_Q shape: (16, 4)")
print("W_K shape: (16, 4)")
print("W_V shape: (16, 4)")
print("  ↓ Weight matrices that project 16-dim → 4-dim")
print()

# After projection
print("Q = X @ W_Q → (5, 16) @ (16, 4) = (5, 4)")
print("K = X @ W_K → (5, 16) @ (16, 4) = (5, 4)")
print("V = X @ W_V → (5, 16) @ (16, 4) = (5, 4)")
print("  ↓ Each token now has 4-dim query, key, value")
print()

# Attention scores
print("scores = Q @ K.T → (5, 4) @ (4, 5) = (5, 5)")
print("  ↓ 5x5 matrix of attention scores")
print()

# After softmax
print("attention_weights = softmax(scores / sqrt(4)) → (5, 5)")
print("  ↓ Each row sums to 1.0 (probabilities)")
print()

# Final output
print("output = attention_weights @ V → (5, 5) @ (5, 4) = (5, 4)")
print("  ↓ 5 tokens, each with 4-dim attended representation")