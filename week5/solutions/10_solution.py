"""SOLUTION — Exercise 10: The 5-Step Training Loop"""
import torch
import torch.nn as nn

torch.manual_seed(42)
X_train = torch.randn(100, 1)
y_train = 2.0 * X_train + 1.0

model     = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(500):
    # Step 1: Forward pass — run inputs through the model
    # model(X) calls model.forward(X) under the hood
    y_pred = model(X_train)

    # Step 2: Compute the loss — Mean Squared Error = mean((y_pred - y_true)^2)
    # This is our measure of "how wrong are we?"
    loss = criterion(y_pred, y_train)

    # Step 3: Zero out gradients from the PREVIOUS step
    # PyTorch ACCUMULATES gradients by default (+=).
    # Without this, old gradients corrupt the new ones.
    optimizer.zero_grad()

    # Step 4: Backprop — compute dL/dW and dL/db via chain rule
    # After this call, model.weight.grad and model.bias.grad are populated
    loss.backward()

    # Step 5: Gradient descent step
    # For each parameter p: p = p - lr * p.grad
    # SGD with lr=0.1 means we take a 10% step in the direction of steepest descent
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1:3d}/500 — loss: {loss.item():.6f}")

print(f"\nW = {model.weight.item():.4f}  (target: 2.0)")
print(f"b = {model.bias.item():.4f}    (target: 1.0)")
print("✅ Solution 10 correct!")
