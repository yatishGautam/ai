"""
EXERCISE 7 [CORE] — Autograd
==============================
Autograd is WHY PyTorch exists.
You define the forward computation. PyTorch builds a computation graph.
When you call .backward(), it walks that graph backwards and computes
every gradient automatically using the chain rule.

You never write a single derivative by hand.

Fill in each TODO, then run: python 7_autograd.py
"""

import torch

print("── EXERCISE 7: Autograd ──────────────────────────────────")

# ── Part 7a: Simple scalar function ──────────────────────────────────────────
# f(x) = x² + 3x + 1
# df/dx = 2x + 3
# At x=4: df/dx = 2(4) + 3 = 11

x = torch.tensor(4.0, requires_grad=True)   # tell PyTorch: track this

# TODO 7a-1: Define y = x² + 3x + 1
# Hint: x**2 + 3*x + 1
y = None  # YOUR CODE HERE

assert y is not None, "Fill in 7a-1!"

# TODO 7a-2: Call backward to compute dy/dx
# Hint: y.backward()
# YOUR CODE HERE

# TODO 7a-3: Print x.grad — should be 11.0
print(f"7a. y = x² + 3x + 1 at x=4")
print(f"    dy/dx = {x.grad}")   # should be tensor(11.)
assert x.grad is not None, "Did you call y.backward()?"
assert abs(x.grad.item() - 11.0) < 1e-4, f"Expected 11.0, got {x.grad.item()}"

# ── Part 7b: Gradient through a linear layer ─────────────────────────────────
# This is exactly what backprop does when training a neural network.
# The gradient tells us: "which direction should we move each weight
# to decrease the loss?"

torch.manual_seed(42)
x = torch.randn(3, requires_grad=True)    # 3 input features
W = torch.randn(3, 2, requires_grad=True) # weight matrix

y    = x @ W       # forward pass through linear layer, shape [2]
loss = y.sum()     # fake scalar loss (sum reduces to scalar for .backward())

# TODO 7b: Call backward to compute all gradients
# After this, W.grad and x.grad will be populated
# Hint: loss.backward()
# YOUR CODE HERE

assert W.grad is not None, "Did you call loss.backward()?"
print(f"\n7b. W.grad shape: {W.grad.shape}")   # should be [3, 2]
print(f"    W.grad:\n{W.grad}")
print(f"    → each entry = how much adjusting that weight affects the loss")
print(f"    → gradient descent update would be: W = W - lr * W.grad")
assert W.grad.shape == torch.Size([3, 2])

# ── Part 7c: torch.no_grad() for inference ───────────────────────────────────
# During training: we NEED gradients to update weights.
# During inference (production, evaluation): we DON'T need them.
# torch.no_grad() skips building the computation graph → faster + less memory.

x = torch.randn(3, requires_grad=True)
W = torch.randn(3, 2, requires_grad=True)

print(f"\n7c. Without no_grad:")
y_train = x @ W
print(f"    y.requires_grad = {y_train.requires_grad}")   # True

# TODO 7c: Redo the same computation inside torch.no_grad()
# and print y_inf.requires_grad — should be False
with torch.no_grad():
    y_inf = None  # YOUR CODE HERE

assert y_inf is not None, "Fill in 7c! Compute x @ W inside the no_grad block."
print(f"    Inside no_grad: y.requires_grad = {y_inf.requires_grad}")  # False
assert y_inf.requires_grad == False, "Inside no_grad, requires_grad should be False"

print("\n✅ Exercise 7 passed!")
