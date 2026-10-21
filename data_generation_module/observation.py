"""Capture variation and count sampling for continuous RNA abundances."""

import numpy as np


def draw_capture_factor(expr_rate, lib_size_mean, lib_size_sd):
    """Draw one capture factor per cell relative to the median total abundance."""
    n_cells = expr_rate.shape[0]
    lib_size = np.random.lognormal(
        mean=np.log(lib_size_mean), sigma=lib_size_sd, size=(n_cells, 1)
    ).astype(np.float32)
    total_abundance = np.clip(expr_rate.sum(axis=1, keepdims=True), 1e-6, None)
    reference_total = np.clip(np.median(total_abundance), 1e-6, None)
    return lib_size / reference_total


def sample_counts(expr_rate, capture_factor, capture_scale, sampling,
                  nb_dispersion, dropout_prob, dropout_slope):
    """Sample spliced or unspliced counts in the specified observation mode."""
    mean_counts = np.clip(expr_rate, 0.0, None) * capture_factor * float(capture_scale)
    if sampling == "poisson":
        counts = np.random.poisson(mean_counts).astype(np.float32)
    elif sampling in ("nb", "negative_binomial"):
        mean_counts_clamped = np.clip(mean_counts, 1e-6, None)
        lam = np.random.gamma(shape=nb_dispersion, scale=mean_counts_clamped / nb_dispersion)
        counts = np.random.poisson(lam).astype(np.float32)
    else:
        counts = mean_counts

    if dropout_prob > 0:
        normalized_expr = mean_counts / (mean_counts.mean(axis=1, keepdims=True) + 1e-6)
        p_dropout = dropout_prob * np.exp(-dropout_slope * normalized_expr)
        dropout_mask = np.random.rand(*counts.shape) < p_dropout
        counts[dropout_mask] = 0
    return counts
