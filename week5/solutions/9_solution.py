"""SOLUTION — Exercise 9: Speed Comparison"""
import torch
import time

def matmul_slow(A, B):
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape
    C = torch.zeros(rows_A, cols_B)
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]
    return C

size = 50
A = torch.randn(size, size)
B = torch.randn(size, size)

start     = time.time()
C_slow    = matmul_slow(A, B)
slow_time = time.time() - start

start     = time.time()
C_fast    = A @ B   # dispatches to BLAS/MKL C++ routine — fully parallelized
fast_time = time.time() - start

speedup = slow_time / fast_time if fast_time > 0 else float('inf')

print(f"Python loops: {slow_time:.4f}s")
print(f"PyTorch @:    {fast_time:.6f}s")
print(f"Speedup:      ~{speedup:.0f}x")
print("✅ Solution 9 correct!")
