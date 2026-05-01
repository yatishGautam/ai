"""
Problem 4.1 — Implement scaled dot-product attention from scratch

This implements the core attention mechanism used in transformers.
The formula is: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

Where:
- Q (Query): What we're looking for
- K (Key): What each position offers
- V (Value): The actual information to retrieve
- d_k: Dimension of queries and keys (used for scaling)
"""

import numpy as np  # NumPy is Python's library for numerical computing


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax along last axis.
    
    Softmax converts numbers into probabilities that sum to 1.
    Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j))
    
    Args:
        x: Input array of any shape
    
    Returns:
        Array of same shape with values between 0 and 1 that sum to 1
    """
    # Subtract max before exp to avoid overflow
    # This is a numerical stability trick: softmax(x) = softmax(x - c) for any constant c
    x_max = np.max(x, axis=-1, keepdims=True)  # Find max along last dimension
    
    # Compute exponentials of (x - max)
    exp_x = np.exp(x - x_max)
    
    # Divide by sum to get probabilities
    # axis=-1 means sum along last dimension
    # keepdims=True preserves dimensions for broadcasting
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray,  # shape: (seq_len, d_k) - Query matrix
    K: np.ndarray,  # shape: (seq_len, d_k) - Key matrix  
    V: np.ndarray,  # shape: (seq_len, d_v) - Value matrix
    mask: np.ndarray = None  # optional causal mask to hide future tokens
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scaled dot-product attention mechanism.
    
    This is the fundamental building block of transformers. It allows each position
    to attend to (focus on) all positions in the input sequence.
    
    Args:
        Q: Query matrix - "what am I looking for?"
        K: Key matrix - "what do I contain?"
        V: Value matrix - "what information do I have?"
        mask: Optional binary mask (1=allow, 0=block) for causal/padding masking
    
    Returns:
        output: Weighted combination of values (seq_len, d_v)
        attention_weights: Attention probabilities (seq_len, seq_len)
    """
    # Get dimension of keys/queries (used for scaling)
    d_k = Q.shape[-1]
    
    # ==========================================
    # Step 1: Compute similarity scores (QK^T)
    # ==========================================
    # This measures how much each query "matches" each key
    # Q @ K.T performs matrix multiplication
    # K.T is the transpose of K (swaps rows and columns)
    # Result shape: (seq_len, seq_len)
    # scores[i, j] = how much position i attends to position j
    scores = Q @ K.T
    
    # ==========================================
    # Step 2: Scale by sqrt(d_k)
    # ==========================================
    # Why scale? Dot products grow with dimension size
    # For large d_k, scores become very large, making softmax saturate
    # Scaling keeps variance around 1, preserving gradient flow
    scores = scores / np.sqrt(d_k)
    
    # ==========================================
    # Step 3: Apply mask if provided
    # ==========================================
    # Masking is used for:
    # - Causal attention: prevent attending to future positions
    # - Padding: ignore padded positions
    # We set masked positions to -infinity so softmax makes them 0
    if mask is not None:
        # np.where(condition, value_if_true, value_if_false)
        # Where mask==0, replace score with -inf
        scores = np.where(mask == 0, -np.inf, scores)
    
    # ==========================================
    # Step 4: Apply softmax to get attention weights
    # ==========================================
    # Softmax converts scores into probabilities
    # Each row will sum to 1.0
    # attention_weights[i, j] = probability that position i attends to position j
    attention_weights = softmax(scores)
    
    # ==========================================
    # Step 5: Weighted sum of values
    # ==========================================
    # Multiply attention weights by values and sum
    # This is like a weighted average of all value vectors
    # Each output position is a mixture of all input values
    # weighted by how much attention we pay to each position
    output = attention_weights @ V  # Shape: (seq_len, d_v)
    
    return output, attention_weights


# ==========================================
# Test the implementation
# ==========================================

# Set random seed for reproducibility (same random numbers each time)
np.random.seed(0)

# Define dimensions
seq_len = 4  # Sequence length (e.g., 4 tokens/words)
d_k = 8      # Dimension of queries and keys
d_v = 8      # Dimension of values

# Create random query, key, value matrices
# np.random.randn generates random numbers from standard normal distribution
Q = np.random.randn(seq_len, d_k)  # Shape: (4, 8)
K = np.random.randn(seq_len, d_k)  # Shape: (4, 8)
V = np.random.randn(seq_len, d_v)  # Shape: (4, 8)

# Run attention
output, weights = scaled_dot_product_attention(Q, K, V)

# Print results
print("Output shape:", output.shape)   # Should be (4, 8)
print("Weights shape:", weights.shape) # Should be (4, 4)
print("Each row sums to:", weights.sum(axis=-1))  # Each row should sum to ~1.0
print("\nAttention weights:\n", np.round(weights, 3))  # Round to 3 decimals

# ==========================================
# Test with causal mask (for autoregressive models like GPT)
# ==========================================
print("\n--- Testing with causal mask ---")
# Causal mask prevents position i from attending to positions j > i
# This is used in language models where we can't see future words
# np.tril creates a lower triangular matrix (1s below diagonal, 0s above)
causal_mask = np.tril(np.ones((seq_len, seq_len)))
print("Causal mask:\n", causal_mask.astype(int))

# Run attention with causal mask
output_masked, weights_masked = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
print("\nCausal attention weights:\n", np.round(weights_masked, 3))
# Notice: upper triangle is now all zeros (can't attend to future)