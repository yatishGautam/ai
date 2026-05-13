"""
EXERCISE 8 [CORE] — Building a Neural Network with nn.Module
=============================================================
nn.Module is the base class for ALL PyTorch models.
You define layers in __init__, and connect them in forward().
PyTorch handles parameter tracking, .grad population, and .parameters() iteration.

This is the MNIST digit classifier architecture from the curriculum.

Fill in each TODO, then run: python 8_neural_network.py
"""

import torch
import torch.nn as nn

print("── EXERCISE 8: Building a Neural Network ─────────────────")

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO 8a: Define the layers
        # Architecture: 784 → 256 → 128 → 10
        #
        # self.layer1 = nn.Linear(784, 256)   ← maps 784 pixel inputs to 256 features
        # self.relu   = nn.ReLU()             ← non-linearity (zeroes out negatives)
        # self.layer2 = nn.Linear(256, 128)   ← maps 256 features to 128
        # self.relu2  = nn.ReLU()
        # self.layer3 = nn.Linear(128, 10)    ← maps 128 features to 10 digit scores
        #
        # YOUR CODE HERE (replace `pass`)
        pass

    def forward(self, x):
        # TODO 8b: Connect the layers in order
        # x → layer1 → relu → layer2 → relu2 → layer3 → return
        #
        # Hint:
        #   x = self.layer1(x)
        #   x = self.relu(x)
        #   ... and so on
        #
        # YOUR CODE HERE (replace `pass`)
        pass

# ── Test the model ────────────────────────────────────────────────────────────
model = MyNet()
test_input = torch.randn(32, 784)   # batch of 32 MNIST images, flattened

try:
    output = model(test_input)
    print(f"8a. Input shape:  {test_input.shape}")
    print(f"    Output shape: {output.shape}")    # should be [32, 10]
    assert output.shape == torch.Size([32, 10]), f"Expected [32,10], got {output.shape}"
    print(f"    10 scores per image — one per digit class!")
except Exception as e:
    print(f"ERROR: {e}")
    print("Check that __init__ defines all layers and forward() connects them.")
    raise

# ── Inspect the weight matrix ─────────────────────────────────────────────────
print(f"\n8b. Layer 1 weight shape: {model.layer1.weight.shape}")   # [256, 784]
print(f"    Layer 1 bias shape:   {model.layer1.bias.shape}")      # [256]
print(f"    The weight matrix IS the learned transformation.")
print(f"    It maps 784 raw pixel values → 256 learned abstract features.")
assert model.layer1.weight.shape == torch.Size([256, 784])
assert model.layer1.bias.shape   == torch.Size([256])

# ── Count parameters ──────────────────────────────────────────────────────────
# TODO 8c: Count the total number of learnable parameters in the model
# Hint: sum(p.numel() for p in model.parameters())
total_params = None   # YOUR CODE HERE

assert total_params is not None, "Fill in 8c!"
print(f"\n8c. Total parameters: {total_params:,}")
# Expected breakdown:
#   layer1: 784*256 + 256 = 200,960
#   layer2: 256*128 + 128 = 32,896
#   layer3: 128*10  + 10  = 1,290
#   Total:                  235,146
assert total_params == 235146, f"Expected 235,146 parameters, got {total_params}"
print(f"    Each is one number that gradient descent adjusts during training.")

print("\n✅ Exercise 8 passed!")
