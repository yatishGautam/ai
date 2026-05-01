"""
Problem 4.2 — Implement a single attention head end-to-end

This builds on Problem 4.1 by adding learnable projection matrices.
Instead of using Q, K, V directly, we learn how to create them from input embeddings.

In real transformers:
- Input embeddings have dimension d_model (e.g., 768 in BERT)
- We project them down to smaller d_k dimensions (e.g., 64)
- This allows multiple heads to run in parallel (multi-head attention)
"""

import numpy as np  # NumPy for numerical computing

# ============================================
# Dependencies from Problem 4.1
# ============================================

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax along last axis.
    Converts scores to probabilities that sum to 1.
    """
    # Subtract max for numerical stability (prevents overflow)
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
    Core attention mechanism from Problem 4.1.
    Returns (output, attention_weights).
    """
    d_k = Q.shape[-1]
    
    # Step 1: Compute similarity scores
    scores = Q @ K.T  # Matrix multiplication: (seq_len, d_k) @ (d_k, seq_len)
    
    # Step 2: Scale to prevent gradient saturation
    scores = scores / np.sqrt(d_k)
    
    # Step 3: Apply mask if provided (e.g., causal mask)
    if mask is not None:
        scores = np.where(mask == 0, -np.inf, scores)
    
    # Step 4: Convert to probabilities
    attention_weights = softmax(scores)
    
    # Step 5: Weighted sum of values
    output = attention_weights @ V
    
    return output, attention_weights


# ============================================
# Problem 4.2 Solution: Attention Head
# ============================================

class AttentionHead:
    """
    A single attention head with learnable projection matrices.
    
    This is one "head" in multi-head attention. It learns to:
    1. Project input embeddings into Query, Key, Value spaces
    2. Apply scaled dot-product attention
    
    In practice, transformers use multiple heads in parallel, each learning
    different attention patterns (e.g., one head might focus on syntax,
    another on semantics).
    """
    
    def __init__(self, d_model: int, d_k: int):
        """
        Initialize the attention head with random weight matrices.
        
        Args:
            d_model: Dimension of input embeddings (e.g., 768 in BERT-base)
            d_k: Dimension to project to for Q, K, V (e.g., 64)
                 Typically d_k = d_model / num_heads
        """
        # Set seed for reproducibility (same weights each time we run)
        np.random.seed(42)
        
        # ==========================================
        # Initialize learnable weight matrices
        # ==========================================
        
        # W_Q: Projects input to Query space
        # Shape: (d_model, d_k)
        # Multiply by 0.1 for small initial weights (helps training stability)
        # np.random.randn generates numbers from standard normal distribution
        self.W_Q = np.random.randn(d_model, d_k) * 0.1
        
        # W_K: Projects input to Key space
        # Shape: (d_model, d_k)
        self.W_K = np.random.randn(d_model, d_k) * 0.1
        
        # W_V: Projects input to Value space
        # Shape: (d_model, d_k)
        # Note: In practice, V can have different dimension d_v,
        # but we use d_k here for simplicity
        self.W_V = np.random.randn(d_model, d_k) * 0.1
        
        # Store d_k for later use
        self.d_k = d_k
    
    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through the attention head.
        
        This is the "forward" pass in neural networks - the computation
        that happens when data flows through the layer.
        
        Args:
            X: Input token embeddings
               Shape: (seq_len, d_model)
               Each row is a d_model-dimensional embedding for one token
        
        Returns:
            output: Attention output (seq_len, d_k)
            attention_weights: Attention probability matrix (seq_len, seq_len)
        """
        
        # ==========================================
        # Step 1: Project X into Query, Key, Value spaces
        # ==========================================
        
        # Create Query matrix by multiplying input with W_Q
        # X @ self.W_Q performs matrix multiplication
        # (seq_len, d_model) @ (d_model, d_k) = (seq_len, d_k)
        # Each row of Q is now a "query" vector asking "what am I looking for?"
        Q = X @ self.W_Q
        
        # Create Key matrix by multiplying input with W_K
        # (seq_len, d_model) @ (d_model, d_k) = (seq_len, d_k)
        # Each row of K is now a "key" vector saying "this is what I contain"
        K = X @ self.W_K
        
        # Create Value matrix by multiplying input with W_V
        # (seq_len, d_model) @ (d_model, d_k) = (seq_len, d_k)
        # Each row of V is now a "value" vector containing actual information
        V = X @ self.W_V
        
        # ==========================================
        # Step 2: Apply scaled dot-product attention
        # ==========================================
        
        # Call the attention function from Problem 4.1
        # This computes: softmax(QK^T / sqrt(d_k)) V
        # Returns the attended output and attention weights
        output, attention_weights = scaled_dot_product_attention(Q, K, V)
        
        # Return both output and weights
        # - output: the context-aware representations (what we use downstream)
        # - attention_weights: for visualization/interpretation
        return output, attention_weights


# ==========================================
# Test the Attention Head
# ==========================================

# Set random seed for reproducibility
np.random.seed(42)

# Define dimensions
seq_len = 5   # Sequence length (e.g., 5 tokens in a sentence)
d_model = 16  # Input embedding dimension (small for testing, real: 768-1024)
d_k = 4       # Projection dimension (small for testing, real: 64)

# Create random input embeddings
# In practice, these would come from a token embedding layer
# Shape: (seq_len, d_model) = (5, 16)
# Each row represents one token's embedding
X = np.random.randn(seq_len, d_model)

# Create an attention head
# This initializes the W_Q, W_K, W_V matrices
head = AttentionHead(d_model, d_k)

# Run forward pass
# This projects X to Q, K, V and computes attention
output, weights = head.forward(X)

# ==========================================
# Print results
# ==========================================

print("Input shape:", X.shape)         # (5, 16) - 5 tokens, 16-dim embeddings
print("Output shape:", output.shape)   # (5, 4) - 5 tokens, 4-dim output
print("\nAttention weights:")
print(np.round(weights, 2))  # Round to 2 decimals for readability

# Verify that attention weights are proper probabilities
print("\nRow sums (should all be ~1.0):", weights.sum(axis=-1))

# ==========================================
# Interpretation of results
# ==========================================
print("\n--- How to read the attention weights ---")
print("Each row i shows how position i attends to all positions.")
print("For example, weights[0] shows how the first token attends to all 5 tokens.")
print("\nHigher values mean stronger attention.")
print("Each row sums to 1.0 (it's a probability distribution).")