"""SOLUTION — Exercise 8: Neural Network with nn.Module"""
import torch
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()   # always call parent __init__ — registers PyTorch internals

        # nn.Linear(in, out) creates a weight matrix [out, in] and bias [out]
        # It implements: output = input @ weight.T + bias
        self.layer1 = nn.Linear(784, 256)   # 784 pixel inputs → 256 hidden features

        # ReLU(x) = max(0, x) — zeroes out negatives, keeps positives
        # Without non-linearity, stacking linear layers = one big linear layer
        self.relu   = nn.ReLU()

        self.layer2 = nn.Linear(256, 128)   # 256 features → 128 features
        self.relu2  = nn.ReLU()

        self.layer3 = nn.Linear(128, 10)    # 128 features → 10 digit class scores

    def forward(self, x):
        # Define the data flow: each call to self.layerN runs that layer
        x = self.layer1(x)   # [batch, 784] → [batch, 256]
        x = self.relu(x)     # apply non-linearity, shape unchanged
        x = self.layer2(x)   # [batch, 256] → [batch, 128]
        x = self.relu2(x)    # apply non-linearity
        x = self.layer3(x)   # [batch, 128] → [batch, 10]
        return x             # raw logits (unnormalized scores), one per digit

model = MyNet()
test_input = torch.randn(32, 784)
output = model(test_input)

print(f"8a. output shape: {output.shape}")                          # [32, 10]
print(f"8b. layer1 weight: {model.layer1.weight.shape}")            # [256, 784]
print(f"    layer1 bias:   {model.layer1.bias.shape}")              # [256]

# model.parameters() iterates over all learnable tensors (all weights and biases)
# .numel() = number of elements in each tensor
# Total: 784*256 + 256 + 256*128 + 128 + 128*10 + 10 = 235,146
total_params = sum(p.numel() for p in model.parameters())

print(f"8c. Total parameters: {total_params:,}")   # 235,146
print("✅ Solution 8 correct!")
