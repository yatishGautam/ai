"""
EXERCISE 5 [CORE] — Dot Product as Similarity
==============================================
The dot product is the atom of deep learning.
Two vectors pointing in the same direction → large positive dot product.
Opposite directions → large negative dot product.
Perpendicular → near zero.

This is EXACTLY how attention decides which tokens to focus on:
  score(query, key) = query · key

Fill in each TODO, then run: python 5_dot_product.py
"""

import torch

print("── EXERCISE 5: Dot Product as Similarity ─────────────────")

# ── Part 5a: Build intuition ──────────────────────────────────────────────────
similar_a = torch.tensor([1.0, 2.0, 3.0])
similar_b = torch.tensor([1.1, 1.9, 3.1])      # almost the same direction
opposite  = torch.tensor([-1.0, -2.0, -3.0])   # exact opposite direction
perp      = torch.tensor([3.0, 0.0, -1.0])     # roughly perpendicular

# TODO 5a-1: Compute dot product of similar_a and similar_b
# Hint: vector1 @ vector2  (works for 1D tensors too)
score_similar = None   # YOUR CODE HERE

# TODO 5a-2: Compute dot product of similar_a and opposite
score_opposite = None  # YOUR CODE HERE

# TODO 5a-3: Compute dot product of similar_a and perp
score_perp = None      # YOUR CODE HERE

assert score_similar  is not None, "Fill in 5a-1!"
assert score_opposite is not None, "Fill in 5a-2!"
assert score_perp     is not None, "Fill in 5a-3!"

print(f"5a. Similar vectors:      {score_similar:.2f}  ← should be HIGH (>10)")
print(f"    Opposite vectors:     {score_opposite:.2f} ← should be NEGATIVE")
print(f"    Perpendicular:        {score_perp:.2f}   ← should be near ZERO")

assert score_similar  > 10,   "Similar vectors should have large positive dot product"
assert score_opposite < 0,    "Opposite vectors should have negative dot product"
assert abs(score_perp) < 5,   "Perpendicular vectors should have small dot product"

# ── Part 5b: Simulate attention ───────────────────────────────────────────────
# "The cat sat on the mat"
# Query = "cat". Which words does "cat" attend to most?
# The word with the highest dot product score gets the most attention weight.

word_vectors = {
    "the":  torch.tensor([0.1,  0.2,  0.3,  0.1]),
    "cat":  torch.tensor([0.9,  0.8, -0.1,  0.7]),   # our query
    "sat":  torch.tensor([0.2,  0.1,  0.8,  0.2]),
    "on":   torch.tensor([0.1,  0.1,  0.1,  0.1]),
    "mat":  torch.tensor([0.8,  0.7, -0.2,  0.6]),   # similar to "cat" — both nouns
}

query = word_vectors["cat"]

print(f"\n5b. Attention scores for 'cat' (higher = more attention):")
scores = {}
for word, vec in word_vectors.items():
    # TODO 5b: Compute dot product of query with each word's vector
    score = None   # YOUR CODE HERE
    assert score is not None, f"Fill in the dot product for '{word}'!"
    scores[word] = score.item()
    print(f"    cat → {word:4s}: {score:.3f}")

most_attended = max(scores, key=scores.get)
print(f"\n    'cat' attends most to: '{most_attended}'")
print(f"    (Makes sense — 'cat' and 'mat' are both concrete nouns!)")

print("\n✅ Exercise 5 passed!")
