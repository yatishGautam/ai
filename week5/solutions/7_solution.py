"""SOLUTION — Exercise 7: Autograd"""
import torch

# requires_grad=True tells PyTorch: "track every operation on this tensor
# so we can compute derivatives later"
x = torch.tensor(4.0, requires_grad=True)

# PyTorch builds a computation graph as you do forward math
# y = x^2 + 3x + 1
y = x**2 + 3*x + 1

# .backward() walks the computation graph backward using the chain rule
# It populates .grad on every leaf tensor that has requires_grad=True
# Here: dy/dx = 2x + 3 = 2(4) + 3 = 11
y.backward()

print(f"7a. dy/dx at x=4: {x.grad}")   # tensor(11.)

# ──────────────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
x = torch.randn(3, requires_grad=True)
W = torch.randn(3, 2, requires_grad=True)

y    = x @ W       # forward: linear layer, shape [2]
loss = y.sum()     # sum to get a scalar (backward requires scalar output)

# Backprop: computes dL/dW and dL/dx simultaneously
# dL/dW[i,j] = dL/dy[j] * x[i]  (outer product, mathematically)
loss.backward()

print(f"\n7b. W.grad shape: {W.grad.shape}")   # [3, 2]
print(f"    W.grad:\n{W.grad}")

# ──────────────────────────────────────────────────────────────────────────────
x = torch.randn(3, requires_grad=True)
W = torch.randn(3, 2, requires_grad=True)

print(f"\n7c. Outside no_grad: requires_grad = {(x @ W).requires_grad}")  # True

# torch.no_grad() disables the computation graph for everything inside the block
# No graph = no memory overhead, no gradient tracking
# Use during inference (production) and evaluation
with torch.no_grad():
    y_inf = x @ W   # same computation, but not tracked

print(f"    Inside  no_grad: requires_grad = {y_inf.requires_grad}")  # False
print("✅ Solution 7 correct!")
