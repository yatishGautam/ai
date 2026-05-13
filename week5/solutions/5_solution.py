"""SOLUTION — Exercise 5: Dot Product as Similarity"""
import torch

similar_a = torch.tensor([1.0, 2.0, 3.0])
similar_b = torch.tensor([1.1, 1.9, 3.1])
opposite  = torch.tensor([-1.0, -2.0, -3.0])
perp      = torch.tensor([3.0, 0.0, -1.0])

# The @ operator works on 1D tensors too — it computes a dot product
# a · b = sum(a[i] * b[i]) for all i
# Geometrically: a · b = |a| * |b| * cos(angle between them)
# Same direction (angle=0):  cos(0)  = 1  → large positive
# Opposite (angle=180):      cos(180) = -1 → negative
# Perpendicular (angle=90):  cos(90)  = 0  → near zero
score_similar  = similar_a @ similar_b    # ≈ 14.06 (vectors nearly parallel)
score_opposite = similar_a @ opposite     # ≈ -14.0 (exact opposite)
score_perp     = similar_a @ perp         # ≈  0.0  (roughly perpendicular)

print(f"Similar:      {score_similar:.2f}")
print(f"Opposite:     {score_opposite:.2f}")
print(f"Perpendicular:{score_perp:.2f}")

word_vectors = {
    "the":  torch.tensor([0.1,  0.2,  0.3,  0.1]),
    "cat":  torch.tensor([0.9,  0.8, -0.1,  0.7]),
    "sat":  torch.tensor([0.2,  0.1,  0.8,  0.2]),
    "on":   torch.tensor([0.1,  0.1,  0.1,  0.1]),
    "mat":  torch.tensor([0.8,  0.7, -0.2,  0.6]),
}
query = word_vectors["cat"]

scores = {}
for word, vec in word_vectors.items():
    # Dot product = similarity score between query ("cat") and this word
    # In real attention: Q @ K.T does this for ALL pairs simultaneously
    score = query @ vec
    scores[word] = score.item()
    print(f"cat → {word:4s}: {score:.3f}")

most = max(scores, key=scores.get)
print(f"\n'cat' attends most to: '{most}'")
print("✅ Solution 5 correct!")
