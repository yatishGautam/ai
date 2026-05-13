# PyTorch Exercises — Intro & Guide

## What This Pack Is

11 exercises that connect everything you've studied so far:

| Already Covered | Used In |
|---|---|
| Tokenization & Embeddings | Ex 2, 4, 5 |
| Transformers & Attention (Q, K, V) | Ex 3, 5, 11 |
| Neural Networks & Backpropagation | Ex 7, 8, 10 |
| Tensors & Matrix Math | Ex 1–6 |

The exercises go in order — each one builds on the last. Don't skip ahead.

---

## How To Run

```bash
# Run a single exercise file
python 1_creating_tensors.py

# Or run them all at once
python 1_creating_tensors.py && python 2_shapes.py && python 3_matmul.py
```

---

## File Map

| File | Exercise | Tag | Concept |
|---|---|---|---|
| `1_creating_tensors.py` | Ex 1 | WARM UP | zeros, ones, randn, arange, linspace |
| `2_shapes.py` | Ex 2 | WARM UP | Reading tensor shapes |
| `3_matmul.py` | Ex 3 | CORE | Matrix multiplication |
| `4_reshaping.py` | Ex 4 | CORE | view() and permute() |
| `5_dot_product.py` | Ex 5 | CORE | Dot product as similarity |
| `6_broadcasting.py` | Ex 6 | CORE | Broadcasting |
| `7_autograd.py` | Ex 7 | CORE | Autograd & gradients |
| `8_neural_network.py` | Ex 8 | CORE | Building with nn.Module |
| `9_speed.py` | Ex 9 | CHALLENGE | Python loops vs PyTorch |
| `10_training_loop.py` | Ex 10 | CHALLENGE | The 5-step training loop |
| `11_attention.py` | Ex 11 | CHALLENGE | Scaled dot-product attention |
| `solutions/` | All | — | Fully commented solutions |

---

## The Core Ideas You're Reinforcing

### 1. Tensors are just arrays with superpowers
- `torch.randn(3, 4)` → 3×4 matrix of random floats
- `.shape` tells you the structure
- `.numel()` tells you total number of elements

### 2. Shape is the first thing you read in any model
When you see `[8, 128, 768]`, read it as:
- **8** → batch size (how many sequences at once)
- **128** → sequence length (tokens per sequence)
- **768** → embedding dimension (features per token)

### 3. Matrix multiplication is the engine
- `A @ B` requires A's last dim == B's first dim
- `[5, 4] @ [4, 3]` → `[5, 3]` ✅
- `[5, 4] @ [5, 4]` → ERROR ❌ (need transpose)
- Every linear layer, every attention score = matrix multiply

### 4. The 5-step training loop (memorize this)
```python
for epoch in range(num_epochs):
    y_pred = model(X)              # 1. forward pass
    loss = criterion(y_pred, y)    # 2. compute loss
    optimizer.zero_grad()          # 3. clear old gradients
    loss.backward()                # 4. backprop — compute new gradients
    optimizer.step()               # 5. update weights
```

### 5. Attention formula (the whole transformer, really)
```
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
```
- `Q @ K.T` → how similar is each query to each key?
- `/ sqrt(d_k)` → prevent scores blowing up as d_k grows
- `softmax(...)` → turn scores into probabilities (sum to 1)
- `@ V` → weighted average of values

---

## What "WARN UP / CORE / CHALLENGE" Means

- **WARM UP** — just filling in one-liners, getting comfortable with syntax
- **CORE** — you need to understand the concept to fill these in
- **CHALLENGE** — multi-step, connects multiple concepts, closest to real code

---

## Common Errors You'll Hit

| Error | Cause | Fix |
|---|---|---|
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | Dimension mismatch in `@` | Check shapes, use `.T` to transpose |
| `AttributeError: 'NoneType' has no attribute 'shape'` | You left a `None` unfilled | Replace the `None` with actual code |
| `RuntimeError: grad can be implicitly created only for scalar outputs` | Calling `.backward()` on non-scalar | Use `.sum().backward()` or `.backward(torch.ones_like(y))` |
| `RuntimeError: Expected all tensors to be on the same device` | Mixing CPU and GPU tensors | Keep everything on CPU for now |

---

## After You Finish

Once all 11 pass, you're ready for:
- Building a full multi-head attention module in PyTorch
- The MiniLLM capstone (Llama-style model on Tiny Shakespeare)
- Understanding transformer source code line by line
