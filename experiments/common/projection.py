"""Explicit projection operators shared by controlled experiments."""

import numpy as np


def positive_cosine_neighbor_projection(v, x, y, indices):
    dx = x[indices] - x[:, None, :]
    den = np.linalg.norm(v, axis=1)[:, None] * np.linalg.norm(dx, axis=2)
    similarity = np.divide(np.einsum("ig,ikg->ik", v, dx), den,
                           out=np.zeros_like(den), where=den > 0)
    weights = np.maximum(similarity, 0.)
    totals = weights.sum(1, keepdims=True)
    weights = np.divide(weights, totals, out=np.zeros_like(weights), where=totals > 0)
    return np.einsum("ik,ikd->id", weights, y[indices] - y[:, None, :])
