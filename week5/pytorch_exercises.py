"""
╔══════════════════════════════════════════════════════════════════╗
║         PYTORCH & TENSORS — CODING EXERCISES                    ║
║         Built specifically for YOUR learning journey            ║
║                                                                  ║
║  What you've studied so far:                                     ║
║  ✅ Tokenization & Embeddings                                    ║
║  ✅ Transformers & Attention (Q, K, V)                           ║
║  ✅ Neural Networks & Backpropagation                            ║
║  ✅ Tensors, Matrices & PyTorch (today)                          ║
║                                                                  ║
║  These exercises CONNECT all of it.                              ║
╚══════════════════════════════════════════════════════════════════╝

HOW TO USE THIS FILE:
- Each exercise has a difficulty tag: [WARM UP] [CORE] [CHALLENGE]
- Each exercise connects to something you've already learned
- Do them IN ORDER — each one builds on the last
- Run the file, see what errors you get, fix them

Run with: python pytorch_exercises.py
"""

import torch
import torch.nn as nn
import time

print("=" * 60)
print("  PYTORCH EXERCISES — Let's solidify everything")
print("=" * 60)


# ══════════════════════════════════════════════════════════════
# EXERCISE 1 [WARM UP]
# Creating Tensors — The Many Ways
# 
# You learned: torch.zeros, ones, randn, arange, linspace
# Goal: Get comfortable with the building blocks
# ══════════════════════════════════════════════════════════════

print("\n── EXERCISE 1: Creating Tensors ──────────────────────────")

def exercise_1():
    """
    Create the following tensors and print their shapes.
    Think about WHY each one is useful in deep learning.
    """

    # TODO 1a: Create a 4x4 matrix of zeros
    # Hint: torch.zeros(...)
    zeros_matrix = None  # replace with your code

    # TODO 1b: Create a 3x3 matrix of ones
    ones_matrix = None  # replace with your code

    # TODO 1c: Create a 5x5 matrix of random values (normal distribution)
    # This is how weight matrices start their life!
    random_matrix = None  # replace with your code

    # TODO 1d: Create a sequence [0, 1, 2, ..., 9]
    # Think: position indices in a transformer
    sequence = None  # replace with your code

    # TODO 1e: Create 6 evenly spaced values between 0 and 1
    # Think: attention weights before softmax
    evenly_spaced = None  # replace with your code

    # Verify your work
    if zeros_matrix is not None:
        print(f"1a. zeros_matrix shape: {zeros_matrix.shape}")        # should be torch.Size([4, 4])
        print(f"1b. ones_matrix shape:  {ones_matrix.shape}")         # should be torch.Size([3, 3])
        print(f"1c. random_matrix shape: {random_matrix.shape}")      # should be torch.Size([5, 5])
        print(f"1d. sequence: {sequence}")                             # should be tensor([0,1,2,...,9])
        print(f"1e. evenly_spaced: {evenly_spaced}")                  # should be 6 values 0 to 1
        print("✅ Exercise 1 done!")
    else:
        print("❌ Exercise 1: Fill in the TODOs above!")

exercise_1()


# ══════════════════════════════════════════════════════════════
# EXERCISE 2 [WARM UP]
# Shape is Everything
#
# You learned: .shape tells you the structure of your data
# Goal: Read shapes the way a pro does
# ══════════════════════════════════════════════════════════════

print("\n── EXERCISE 2: Shape is Everything ──────────────────────")

def exercise_2():
    """
    Given these tensors, answer what each dimension MEANS.
    This is the most important skill for reading model code.
    """

    # These are real shapes from transformer models
    batch_size = 8
    seq_len    = 128
    embed_dim  = 768
    num_heads  = 12
    head_dim   = embed_dim // num_heads  # = 64

    t1 = torch.randn(batch_size, seq_len, embed_dim)
    t2 = torch.randn(batch_size, seq_len, num_heads, head_dim)
    t3 = torch.randn(batch_size, num_heads, seq_len, head_dim)
    t4 = torch.randn(num_heads, seq_len, seq_len)

    print(f"t1 shape: {t1.shape}")
    print(f"t2 shape: {t2.shape}")
    print(f"t3 shape: {t3.shape}")
    print(f"t4 shape: {t4.shape}")

    # TODO: Fill in the blanks — what does each dimension mean?
    #
    # t1 = [8, 128, 768]
    #       ^    ^    ^
    #       |    |    └── ??? (what is 768?)
    #       |    └─────── ??? (what is 128?)
    #       └──────────── ??? (what is 8?)
    #
    # t4 = [12, 128, 128]
    #       ^    ^    ^
    #       |    |    └── ??? 
    #       |    └─────── ??? 
    #       └──────────── ??? 
    #
    # HINT: t4 looks like an attention score matrix!
    # Each head has a 128x128 grid — why 128x128?
    # Because each token attends to every other token!

    # TODO 2: How many total numbers are stored in t1?
    # Hint: multiply all dimensions together OR use .numel()
    total_numbers = None  # replace with your code
    
    if total_numbers is not None:
        print(f"\nTotal numbers in t1: {total_numbers}")  # should be 786432
        print("✅ Exercise 2 done!")
    else:
        print("❌ Exercise 2: Calculate total_numbers!")

exercise_2()


# ══════════════════════════════════════════════════════════════
# EXERCISE 3 [CORE]
# Matrix Multiplication — The Engine
#
# You learned: @ operator, inner dimensions must match
# You studied: attention = Q @ K.T, output = W @ input
# Goal: Do matmul confidently, understand the shapes
# ══════════════════════════════════════════════════════════════

print("\n── EXERCISE 3: Matrix Multiplication ────────────────────")

def exercise_3():
    """
    Matrix multiplication is the core of EVERYTHING.
    Let's practice it with real AI shapes.
    """

    # TODO 3a: Multiply these two matrices
    # Think: a single linear layer in a neural network
    # Input: 5 data points, each with 4 features
    # Weight: transforms 4 features into 3 outputs
    X = torch.randn(5, 4)   # 5 samples, 4 features
    W = torch.randn(4, 3)   # weight matrix

    result = None  # TODO: X @ W
    
    if result is not None:
        print(f"3a. X shape: {X.shape}, W shape: {W.shape}")
        print(f"    result shape: {result.shape}")  # should be [5, 3]
        print(f"    Meaning: 5 samples, each now has 3 features (transformed!)")

    # TODO 3b: Attention scores — Q @ K.T
    # This is EXACTLY what happens in attention!
    # Remember from your transformer studies: scores = Q @ K.T
    seq_len = 6
    d_k = 64

    Q = torch.randn(seq_len, d_k)   # 6 tokens, 64-dim queries
    K = torch.randn(seq_len, d_k)   # 6 tokens, 64-dim keys

    # TODO: Compute attention scores
    # Hint: you need to TRANSPOSE K (K.T)
    scores = None  # replace with your code

    if scores is not None:
        print(f"\n3b. Q shape: {Q.shape}, K.T shape: {K.T.shape}")
        print(f"    scores shape: {scores.shape}")  # should be [6, 6]
        print(f"    Meaning: each of 6 tokens has a score for every other token")

    # TODO 3c: Why does this FAIL? Fix it.
    A = torch.randn(3, 4)
    B = torch.randn(3, 4)
    try:
        broken = A @ B  # This will throw an error! WHY?
        print(f"\n3c. result: {broken.shape}")
    except RuntimeError as e:
        print(f"\n3c. ERROR (expected): {e}")
        print(f"    FIX: to multiply A @ B, the inner dims must match.")
        print(f"    A is (3,4) and B is (3,4)... what needs to change?")
        # TODO: Fix this multiplication using transpose
        fixed = None  # A @ B.T  or  A.T @ B  — which one makes sense?
        if fixed is not None:
            print(f"    Fixed result shape: {fixed.shape}")

    print("✅ Exercise 3 done!")

exercise_3()


# ══════════════════════════════════════════════════════════════
# EXERCISE 4 [CORE]
# Reshaping — view() and permute()
#
# You learned: view() changes shape without copying data
#              permute() swaps axes
# Goal: Reshape tensors the way a transformer does
# ══════════════════════════════════════════════════════════════

print("\n── EXERCISE 4: Reshaping ─────────────────────────────────")

def exercise_4():
    """
    Reshaping is what allows one tensor to be interpreted
    multiple different ways — crucial for multi-head attention.
    """

    # TODO 4a: Flatten a batch of images
    # Neural networks often need flat vectors, not 2D grids
    batch_of_images = torch.randn(64, 28, 28)  # 64 MNIST images, 28x28 pixels
    
    # TODO: Reshape to [64, 784] (flatten each image to a vector)
    # Hint: 28 * 28 = 784
    flattened = None  # replace with your code

    if flattened is not None:
        print(f"4a. Before: {batch_of_images.shape}")
        print(f"    After:  {flattened.shape}")  # should be [64, 784]
        print(f"    Same memory? {batch_of_images.data_ptr() == flattened.data_ptr()}")  # True!

    # TODO 4b: Split embeddings into attention heads
    # This is the EXACT operation inside every transformer!
    batch_size = 8
    seq_len    = 128
    embed_dim  = 768
    num_heads  = 12
    head_dim   = 64

    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # TODO: Reshape x to [8, 128, 12, 64]
    # Hint: .view(batch_size, seq_len, num_heads, head_dim)
    x_heads = None  # replace with your code

    if x_heads is not None:
        print(f"\n4b. Before: {x.shape}")
        print(f"    After:  {x_heads.shape}")  # should be [8, 128, 12, 64]

    # TODO 4c: Permute so heads come BEFORE tokens
    # Each head needs to see ALL tokens — that's why heads must be dim 1
    if x_heads is not None:
        # TODO: permute from [8, 128, 12, 64] to [8, 12, 128, 64]
        # Hint: .permute(0, 2, 1, 3)
        x_ready = None  # replace with your code

        if x_ready is not None:
            print(f"\n4c. Before permute: {x_heads.shape}")
            print(f"    After permute:  {x_ready.shape}")  # should be [8, 12, 128, 64]
            print(f"    This is EXACTLY what happens inside every transformer!")

    print("✅ Exercise 4 done!")

exercise_4()


# ══════════════════════════════════════════════════════════════
# EXERCISE 5 [CORE]
# The Dot Product — Similarity in Action
#
# You learned: dot product = similarity
# You studied: attention scores = Q · K
# Goal: SEE similarity numerically, connect it to attention
# ══════════════════════════════════════════════════════════════

print("\n── EXERCISE 5: Dot Product as Similarity ─────────────────")

def exercise_5():
    """
    The dot product is the atom of deep learning.
    Let's build intuition for what it means.
    """

    # TODO 5a: Compute these dot products and explain the result
    similar_a = torch.tensor([1.0, 2.0, 3.0])
    similar_b = torch.tensor([1.1, 1.9, 3.1])    # almost the same direction
    opposite  = torch.tensor([-1.0, -2.0, -3.0]) # exact opposite
    perp      = torch.tensor([3.0, 0.0, -1.0])   # perpendicular (roughly)

    # TODO: Compute dot products using the @ operator
    score_similar  = None  # similar_a @ similar_b
    score_opposite = None  # similar_a @ opposite
    score_perp     = None  # similar_a @ perp

    if score_similar is not None:
        print(f"5a. Similar vectors dot product:    {score_similar:.2f}  ← should be HIGH")
        print(f"    Opposite vectors dot product:   {score_opposite:.2f} ← should be NEGATIVE")
        print(f"    Perpendicular dot product:      {score_perp:.2f}   ← should be near ZERO")

    # TODO 5b: Simulate attention
    # "The cat sat on the mat" — which words relate to "cat"?
    # We'll fake word vectors to demonstrate
    torch.manual_seed(42)
    
    word_vectors = {
        "the":  torch.tensor([0.1,  0.2,  0.3,  0.1]),
        "cat":  torch.tensor([0.9,  0.8, -0.1,  0.7]),  # our query
        "sat":  torch.tensor([0.2,  0.1,  0.8,  0.2]),
        "on":   torch.tensor([0.1,  0.1,  0.1,  0.1]),
        "mat":  torch.tensor([0.8,  0.7, -0.2,  0.6]),  # similar to "cat" (both nouns)
    }

    query = word_vectors["cat"]  # "cat" asks: who should I attend to?

    print(f"\n5b. Attention scores for 'cat':")
    scores = {}
    for word, vec in word_vectors.items():
        # TODO: compute dot product of query with each word vector
        score = None  # query @ vec
        if score is not None:
            scores[word] = score.item()
            print(f"    cat → {word}: {score:.3f}")

    if scores:
        most_attended = max(scores, key=scores.get)
        print(f"\n    'cat' attends most to: '{most_attended}'")
        print(f"    (Makes sense — both 'cat' and 'mat' are concrete nouns!)")

    print("✅ Exercise 5 done!")

exercise_5()


# ══════════════════════════════════════════════════════════════
# EXERCISE 6 [CORE]
# Broadcasting — PyTorch's Secret Weapon
#
# You learned: PyTorch stretches smaller tensors to match larger ones
# Goal: Understand when and why broadcasting happens
# ══════════════════════════════════════════════════════════════

print("\n── EXERCISE 6: Broadcasting ──────────────────────────────")

def exercise_6():
    """
    Broadcasting is used ALL THE TIME in deep learning.
    Adding bias vectors, scaling attention, normalizing — all broadcasting.
    """

    # TODO 6a: Add a bias vector to a batch of outputs
    # This is exactly what happens after nn.Linear!
    batch_output = torch.zeros(32, 10)   # 32 samples, 10 class scores
    bias         = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.6, 0.7, 0.8, 0.9, 1.0])  # shape [10]

    # TODO: Add bias to every row of batch_output
    result = None  # batch_output + bias

    if result is not None:
        print(f"6a. batch_output: {batch_output.shape}")
        print(f"    bias:          {bias.shape}")
        print(f"    result:        {result.shape}")    # should be [32, 10]
        print(f"    First row: {result[0]}")           # should match bias values

    # TODO 6b: Scale attention scores
    # In transformers, attention scores are scaled by sqrt(d_k)
    # to prevent them from becoming too large (which kills gradients)
    import math
    d_k = 64
    attention_scores = torch.randn(8, 12, 128, 128)  # [batch, heads, seq, seq]
    
    # TODO: Divide all scores by sqrt(d_k)
    # Hint: attention_scores / math.sqrt(d_k)
    scaled_scores = None  # replace with your code

    if scaled_scores is not None:
        print(f"\n6b. Before scaling — std: {attention_scores.std():.3f}")
        print(f"    After scaling  — std: {scaled_scores.std():.3f}")
        print(f"    Values are smaller now → softmax won't saturate → better gradients!")

    print("✅ Exercise 6 done!")

exercise_6()


# ══════════════════════════════════════════════════════════════
# EXERCISE 7 [CORE]
# Autograd — PyTorch Computes Gradients For You
#
# You learned: requires_grad=True, .backward(), .grad
# You studied: backpropagation in neural networks
# Goal: See autograd work on a real example
# ══════════════════════════════════════════════════════════════

print("\n── EXERCISE 7: Autograd ──────────────────────────────────")

def exercise_7():
    """
    Autograd is WHY PyTorch exists.
    You define forward. PyTorch handles backward.
    """

    # TODO 7a: Simple gradient — f(x) = x² + 3x + 1
    # The derivative is: df/dx = 2x + 3
    # At x=4: df/dx = 2(4) + 3 = 11
    x = torch.tensor(4.0, requires_grad=True)

    # TODO: Define the function y = x² + 3x + 1
    y = None  # replace with your code

    if y is not None:
        # TODO: Compute gradients
        # Hint: y.backward()
        # TODO: Print x.grad (should be 11.0)
        pass  # replace with your code

    # TODO 7b: Gradient through a neural network layer
    # This is literally what backprop does!
    torch.manual_seed(42)
    
    x   = torch.randn(3, requires_grad=True)   # 3 input features
    W   = torch.randn(3, 2, requires_grad=True) # weight matrix
    
    # Forward pass: linear layer
    y = x @ W         # shape [2]
    loss = y.sum()    # fake loss (sum everything)

    # TODO: Call backward to compute all gradients
    # Then print W.grad — the gradient tells us how to update W
    # Hint: loss.backward()
    pass  # replace with your code

    if W.grad is not None:
        print(f"\n7b. W.grad shape: {W.grad.shape}")  # should be [3, 2]
        print(f"    W.grad: {W.grad}")
        print(f"    This tells us how much to adjust each weight!")
        print(f"    In training: W = W - lr * W.grad")

    # TODO 7c: Understand torch.no_grad()
    # When we're NOT training (inference / evaluation), we don't need gradients
    x = torch.randn(3, requires_grad=True)
    W = torch.randn(3, 2, requires_grad=True)

    print(f"\n7c. Without no_grad:")
    y = x @ W
    print(f"    y.requires_grad = {y.requires_grad}")  # True — tracking!

    # TODO: Do the same calculation inside torch.no_grad()
    # and show that requires_grad becomes False
    with torch.no_grad():
        pass  # replace: y_inf = x @ W, then print y_inf.requires_grad

    print("✅ Exercise 7 done!")

exercise_7()


# ══════════════════════════════════════════════════════════════
# EXERCISE 8 [CORE]
# Building a Neural Network with nn.Module
#
# You learned: define layers in __init__, connect in forward
# You studied: neural networks from class 2
# Goal: Build a real PyTorch model from scratch
# ══════════════════════════════════════════════════════════════

print("\n── EXERCISE 8: Building a Neural Network ─────────────────")

def exercise_8():
    """
    Put it all together — build a simple neural network.
    This is the exact architecture from your MNIST example.
    """

    # TODO 8a: Build a 3-layer network
    # Input: 784 (MNIST flat image)
    # Hidden: 256 neurons
    # Output: 10 (digits 0-9)
    
    class MyNet(nn.Module):
        def __init__(self):
            super().__init__()
            # TODO: Define 3 layers
            # layer1: nn.Linear(784, 256)
            # relu:   nn.ReLU()
            # layer2: nn.Linear(256, 128)
            # relu2:  nn.ReLU()
            # layer3: nn.Linear(128, 10)
            pass  # replace with your code

        def forward(self, x):
            # TODO: Connect the layers
            # x → layer1 → relu → layer2 → relu2 → layer3 → return
            pass  # replace with your code

    # Test it
    model = MyNet()
    test_input = torch.randn(32, 784)  # batch of 32 images

    try:
        output = model(test_input)
        print(f"8a. Input shape:  {test_input.shape}")
        print(f"    Output shape: {output.shape}")   # should be [32, 10]
        print(f"    10 scores per image — one per digit!")
        
        # TODO 8b: Peek inside — what IS a weight matrix?
        print(f"\n8b. Layer 1 weight shape: {model.layer1.weight.shape}")  # [256, 784]
        print(f"    Layer 1 bias shape:   {model.layer1.bias.shape}")    # [256]
        print(f"    This weight matrix IS the transformation!")
        print(f"    It maps 784 pixel values → 256 learned features")

        # TODO 8c: Count total parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n8c. Total parameters: {total_params:,}")
        print(f"    Each parameter is one number that gets adjusted during training")
    except Exception as e:
        print(f"8a. Complete the __init__ and forward methods first! Error: {e}")

    print("✅ Exercise 8 done!")

exercise_8()


# ══════════════════════════════════════════════════════════════
# EXERCISE 9 [CHALLENGE]
# Speed: Python Loops vs PyTorch
#
# You learned: GPU parallelism, embarrassingly parallel
# Goal: FEEL the performance difference yourself
# ══════════════════════════════════════════════════════════════

print("\n── EXERCISE 9: Speed Comparison ──────────────────────────")

def exercise_9():
    """
    This is the GPU argument made viscerally real.
    Compare pure Python loops vs PyTorch's matmul.
    """

    def matmul_slow(A, B):
        """Pure Python matrix multiplication — the painful way"""
        rows_A, cols_A = A.shape
        rows_B, cols_B = B.shape
        C = torch.zeros(rows_A, cols_B)
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    C[i][j] += A[i][k] * B[k][j]
        return C

    size = 50   # start small — loops are SLOW
    A = torch.randn(size, size)
    B = torch.randn(size, size)

    # Time the slow way
    start = time.time()
    C_slow = matmul_slow(A, B)
    slow_time = time.time() - start

    # TODO: Time the fast way (A @ B)
    start = time.time()
    C_fast = None  # replace: A @ B
    fast_time = time.time() - start

    if C_fast is not None:
        speedup = slow_time / fast_time if fast_time > 0 else float('inf')
        print(f"9. Matrix size: {size}x{size}")
        print(f"   Python loops: {slow_time:.4f}s")
        print(f"   PyTorch @:    {fast_time:.6f}s")
        print(f"   Speedup:      {speedup:.0f}x faster!")
        print(f"")
        print(f"   Now imagine size=4096 (actual GPT weights)...")
        print(f"   Loops would take hours. PyTorch takes milliseconds.")
        print(f"   THIS is why GPUs and PyTorch exist.")

    print("✅ Exercise 9 done!")

exercise_9()


# ══════════════════════════════════════════════════════════════
# EXERCISE 10 [CHALLENGE]
# The 5-Line Training Loop
#
# You learned: forward → loss → zero_grad → backward → step
# You studied: backpropagation from class 2
# Goal: Train a model to learn y = 2x + 1 from scratch
# ══════════════════════════════════════════════════════════════

print("\n── EXERCISE 10: The Training Loop ────────────────────────")

def exercise_10():
    """
    The crown jewel. 
    Train a tiny model to learn a simple function.
    
    We'll teach it: y = 2x + 1
    The model starts with random weights.
    After training, it should discover W≈2.0, b≈1.0 on its own.
    """

    # Generate training data: y = 2x + 1
    torch.manual_seed(42)
    X_train = torch.randn(100, 1)               # 100 random x values
    y_train = 2.0 * X_train + 1.0              # the true relationship

    # A single linear layer (our "model")
    # It will learn: y = W*x + b
    # We want it to discover W=2.0 and b=1.0
    model = nn.Linear(1, 1)
    
    print(f"10. Before training:")
    print(f"    W = {model.weight.item():.4f} (random, should become ~2.0)")
    print(f"    b = {model.bias.item():.4f}   (random, should become ~1.0)")

    # TODO: Set up loss function and optimizer
    criterion = nn.MSELoss()                            # Mean Squared Error
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # Stochastic Gradient Descent

    # TODO: The training loop!
    # For each epoch:
    #   1. y_pred = model(X_train)           ← forward pass
    #   2. loss = criterion(y_pred, y_train)  ← compute loss
    #   3. optimizer.zero_grad()             ← clear old gradients
    #   4. loss.backward()                   ← compute new gradients
    #   5. optimizer.step()                  ← update weights
    
    for epoch in range(500):
        # TODO: Fill in the 5 steps
        pass  # replace with your code

    print(f"\n    After training:")
    print(f"    W = {model.weight.item():.4f} (should be close to 2.0)")
    print(f"    b = {model.bias.item():.4f}   (should be close to 1.0)")
    print(f"")
    print(f"    The model DISCOVERED the relationship y = 2x + 1")
    print(f"    entirely from data — no one told it the formula!")

    print("✅ Exercise 10 done!")

exercise_10()


# ══════════════════════════════════════════════════════════════
# EXERCISE 11 [CHALLENGE — CONNECTS EVERYTHING]
# Mini Scaled Dot-Product Attention in PyTorch
#
# You studied: attention from transformers class
# You just learned: matmul, softmax, reshaping
# Goal: Implement attention using PyTorch (upgrade your numpy version)
# ══════════════════════════════════════════════════════════════

print("\n── EXERCISE 11: Attention in PyTorch ─────────────────────")

def exercise_11():
    """
    You already implemented attention in NumPy.
    Now do it in PyTorch.
    
    The formula:  Attention(Q, K, V) = softmax(QK^T / √d_k) × V
    
    This is the formula that powers every transformer —
    GPT-4, BERT, Claude — all of them.
    """

    def scaled_dot_product_attention(Q, K, V):
        """
        Q: shape [seq_len, d_k]
        K: shape [seq_len, d_k]
        V: shape [seq_len, d_v]
        
        Returns: output [seq_len, d_v], weights [seq_len, seq_len]
        """
        import math
        d_k = Q.shape[-1]

        # TODO Step 1: Compute raw attention scores
        # scores = Q @ K.T
        # Shape: [seq_len, seq_len]
        scores = None  # replace with your code

        # TODO Step 2: Scale by sqrt(d_k)
        # Without scaling, large d_k causes scores to blow up
        # and softmax becomes saturated (gradients die)
        scores = None  # replace: scores / math.sqrt(d_k)

        # TODO Step 3: Softmax over last dimension
        # This turns raw scores into probabilities (sum to 1)
        weights = None  # replace: torch.softmax(scores, dim=-1)

        # TODO Step 4: Weighted sum of V
        # Each token's output = weighted combination of ALL value vectors
        output = None  # replace: weights @ V

        return output, weights

    # Test it
    torch.manual_seed(0)
    seq_len, d_k, d_v = 5, 8, 8

    Q = torch.randn(seq_len, d_k)
    K = torch.randn(seq_len, d_k)
    V = torch.randn(seq_len, d_v)

    output, weights = scaled_dot_product_attention(Q, K, V)
    
    if output is not None:
        print(f"11. output shape:  {output.shape}")   # should be [5, 8]
        print(f"    weights shape: {weights.shape}")  # should be [5, 5]
        print(f"    Row sums (should all be 1.0): {weights.sum(dim=-1)}")
        print(f"")
        print(f"    You just implemented the core of GPT/BERT/Claude!")
        print(f"    This exact function runs billions of times during training.")

    print("✅ Exercise 11 done!")

exercise_11()


# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  SUMMARY OF WHAT YOU PRACTICED:")
print("=" * 60)
print("""
  Ex 1  — Creating tensors (zeros, ones, randn, arange, linspace)
  Ex 2  — Reading shapes like a pro
  Ex 3  — Matrix multiplication & understanding shapes
  Ex 4  — Reshaping (view + permute) — what transformers do
  Ex 5  — Dot product as similarity — the atom of attention
  Ex 6  — Broadcasting — adding bias, scaling scores
  Ex 7  — Autograd — automatic backpropagation
  Ex 8  — Building a neural network with nn.Module
  Ex 9  — Speed: loops vs PyTorch (feel the difference)
  Ex 10 — The 5-line training loop (train a model from scratch)
  Ex 11 — Scaled dot-product attention in PyTorch

  Complete all 11 → you understand the full stack from
  raw tensors → matrix math → attention → training loops.
""")
