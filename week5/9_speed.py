"""
EXERCISE 9 [CHALLENGE] — Speed: Python Loops vs PyTorch
========================================================
Python loops = sequential, one multiply at a time.
PyTorch @ = parallelized C++/BLAS under the hood, GPU-ready.

The difference becomes exponentially larger as matrix size grows.
GPT-3's weight matrices are 12,288 × 12,288.
At that size, a Python loop would take hours; PyTorch takes milliseconds.

Fill in the TODO, then run: python 9_speed.py
"""

import torch
import time

print("── EXERCISE 9: Speed Comparison ──────────────────────────")

def matmul_slow(A, B):
    """
    Pure Python triple-nested loop matrix multiplication.
    O(n³) with terrible constant factor — every multiply is a Python call.
    """
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape
    C = torch.zeros(rows_A, cols_B)
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]
    return C

size = 50   # keep small — the loop version is painful even at this size
A = torch.randn(size, size)
B = torch.randn(size, size)

# Time the slow Python loop version
print(f"Running {size}×{size} matrix multiply...")
start    = time.time()
C_slow   = matmul_slow(A, B)
slow_time = time.time() - start

# TODO 9: Compute A @ B using PyTorch and time it
start  = time.time()
C_fast = None   # YOUR CODE HERE
fast_time = time.time() - start

assert C_fast is not None, "Fill in the TODO — compute A @ B!"

# Verify correctness: both methods should give same result
assert torch.allclose(C_slow, C_fast, atol=1e-4), "Results don't match — check your matmul"

speedup = slow_time / fast_time if fast_time > 0 else float('inf')

print(f"\nMatrix size: {size}×{size}")
print(f"Python loops: {slow_time:.4f}s")
print(f"PyTorch @:    {fast_time:.6f}s")
print(f"Speedup:      ~{speedup:.0f}x faster")
print()
print("Now imagine size=4096 (actual GPT-2 weight matrix size)...")
print("Python loop: ~hours. PyTorch: ~milliseconds.")
print("THIS is why GPUs, BLAS, and PyTorch exist.")

print("\n✅ Exercise 9 passed!")
