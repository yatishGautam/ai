import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along last axis."""
    # Subtract max before exp to avoid overflow
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def scaled_dot_product_attention(
    Q: np.ndarray,  # shape: (seq_len, d_k)
    K: np.ndarray,  # shape: (seq_len, d_k)
    V: np.ndarray,  # shape: (seq_len, d_v)
    mask: np.ndarray = None  # optional causal mask
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
    - output: shape (seq_len, d_v) — context-aware representations
    - attention_weights: shape (seq_len, seq_len) — for visualization
    """
    d_k = Q.shape[-1]
    
    # Step 1: compute similarity scores (QK^T)
    scores = Q @ K.T  # (seq_len, seq_len)
    
    # Step 2: scale by sqrt(d_k) to prevent softmax saturation
    scores = scores / np.sqrt(d_k)
    
    # Step 3: apply mask if provided (set masked positions to -inf)
    if mask is not None:
        scores = np.where(mask == 0, -np.inf, scores)
    
    # Step 4: softmax to get attention weights
    attention_weights = softmax(scores)  # (seq_len, seq_len)
    
    # Step 5: weighted sum of V
    output = attention_weights @ V  # (seq_len, d_v)
    
    return output, attention_weights

# Test it
np.random.seed(0)
seq_len, d_k, d_v = 4, 8, 8
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_v)

output, weights = scaled_dot_product_attention(Q, K, V)
print("Output shape:", output.shape)   # (4, 8)
print("Weights shape:", weights.shape) # (4, 4)
print("Each row sums to:", weights.sum(axis=-1))  # all ~1.0
print("\nAttention weights:\n", np.round(weights, 3))

# Test with causal mask
print("\n--- Testing with causal mask ---")
causal_mask = np.tril(np.ones((seq_len, seq_len)))  # lower triangular
output_masked, weights_masked = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
print("Causal attention weights:\n", np.round(weights_masked, 3))