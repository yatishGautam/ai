"""
EXERCISE 10 [CHALLENGE] — The 5-Step Training Loop
===================================================
This is the crown jewel.

We'll train a model to learn the function: y = 2x + 1
It starts with random weights. After 500 gradient descent steps,
it should discover W ≈ 2.0 and b ≈ 1.0 entirely from data.

The 5 steps — memorize these, they never change:
  1. y_pred = model(X)              ← forward pass
  2. loss = criterion(y_pred, y)    ← how wrong are we?
  3. optimizer.zero_grad()          ← clear stale gradients from last step
  4. loss.backward()                ← compute new gradients (backprop)
  5. optimizer.step()               ← nudge weights in the right direction

Fill in the TODO, then run: python 10_training_loop.py
"""

import torch
import torch.nn as nn

print("── EXERCISE 10: The Training Loop ────────────────────────")

torch.manual_seed(42)

# Generate data: y = 2x + 1 with no noise
X_train = torch.randn(100, 1)
y_train = 2.0 * X_train + 1.0

# Single linear layer — it will learn W and b on its own
model     = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

print("Before training:")
print(f"  W = {model.weight.item():.4f}  (random — should converge to ~2.0)")
print(f"  b = {model.bias.item():.4f}    (random — should converge to ~1.0)")

# TODO: Fill in the 5-step training loop
for epoch in range(500):
    # Step 1: Forward pass — run the input through the model
    y_pred = None   # YOUR CODE HERE: model(X_train)

    # Step 2: Compute loss — how far off are our predictions?
    loss = None     # YOUR CODE HERE: criterion(y_pred, y_train)

    # Step 3: Zero gradients — MUST do this before backward()
    #         Without it, gradients accumulate across steps (wrong!)
    # YOUR CODE HERE: optimizer.zero_grad()

    # Step 4: Backward pass — compute gradients via backprop
    # YOUR CODE HERE: loss.backward()

    # Step 5: Update weights — gradient descent step
    # YOUR CODE HERE: optimizer.step()

    # Optional: print progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        if loss is not None:
            print(f"  Epoch {epoch+1:3d}/500 — loss: {loss.item():.6f}")

# ── Verify convergence ────────────────────────────────────────────────────────
print(f"\nAfter training:")
print(f"  W = {model.weight.item():.4f}  (should be close to 2.0)")
print(f"  b = {model.bias.item():.4f}    (should be close to 1.0)")

assert abs(model.weight.item() - 2.0) < 0.05, \
    f"W didn't converge! Got {model.weight.item():.4f}, expected ~2.0"
assert abs(model.bias.item() - 1.0) < 0.05, \
    f"b didn't converge! Got {model.bias.item():.4f}, expected ~1.0"

print("\n  The model DISCOVERED y = 2x + 1 from data alone.")
print("  No one told it the formula. Gradient descent found it.")

print("\n✅ Exercise 10 passed!")
