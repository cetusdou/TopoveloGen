"""Coordinate-aware gene-space velocity metrics."""

import numpy as np


def cosine(a, b, relative_floor=1e-6):
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    fa = relative_floor * np.median(na[na > 0]) if np.any(na > 0) else np.inf
    fb = relative_floor * np.median(nb[nb > 0]) if np.any(nb > 0) else np.inf
    valid = np.isfinite(a).all(1) & np.isfinite(b).all(1) & (na > fa) & (nb > fb)
    out = np.full(len(a), np.nan)
    out[valid] = np.einsum("ij,ij->i", a[valid] / na[valid, None], b[valid] / nb[valid, None])
    return np.clip(out, -1, 1)


def pushforward(s, v, target=10000.):
    total = s.sum(1, keepdims=True)
    vp = target * (v - s * v.sum(1, keepdims=True) / total) / total
    return vp, vp / (1 + target * s / total)
