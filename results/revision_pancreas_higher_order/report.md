# R1 Major 14: pancreas higher-order count statistics

Held-out metrics, **not a held-out dataset**. The same pancreas reference guided simulator topology and marginal calibration. No gene identity correspondence exists between synthetic and real matrices; genes were abundance-quantile matched. These are descriptive comparisons, not independent generalization estimates.

Current-code Figure 4 Stage 2 replay from cache; 8 deterministic draws of 512 cells and 256 genes per group. Full provenance and exact indices: `manifest.json`.

| Metric | Real median | Synthetic median | Synthetic - real | W1 | Across-draw difference range |
|---|---:|---:|---:|---:|---:|
| u_s_raw | 0.00134 | -0.00795 | -0.00929 | 0.0159 | [-0.0286, -0.00422] |
| u_s_normalized | -0.00166 | -0.00955 | -0.00789 | 0.0169 | [-0.0258, -0.00518] |
| conditional_fano_spliced | 1.12 | 1.04 | -0.0776 | 0.169 | [-0.0776, -0.00493] |
| gene_covariance_log_cpm | 0 | -0.00275 | -0.00275 | 0.0308 | [-0.00281, -0.00167] |
| gene_correlation_log_cpm | -0.00521 | -0.00737 | -0.00216 | 0.00427 | [-0.00216, 6.77e-05] |
| cell_distance_log_cpm | 1.81 | 1.89 | 0.0853 | 0.0819 | [0.0477, 0.135] |
| cell_distance_gene_z | 1.01 | 1.38 | 0.371 | 0.319 | [0.35, 0.381] |

W1 is the one-dimensional Wasserstein distance. Ranges reflect sample sensitivity only; correlated gene/cell pairs preclude naive pair-bootstrap confidence intervals. Raw-count conditional Fano conditions on library-size quartiles but may retain mean-expression and cell-state confounding. Log-normalization addresses cell-depth scale; gene-z distances probe shape after marginal scale removal. Neither is a fitted calibration correction.

## Joint spliced/unspliced counts

The two PDF heatmaps use the same first prespecified draw: 131,072 same-cell/same-gene (log1p(u), log1p(s)) pairs per group. Their coarse common-bin total variation is **0.025**; across the same eight draws it ranges [0.025, 0.030]. Excluding (0,0) and renormalizing the remaining pairs gives **0.216** (range [0.211, 0.254]), exposing shape differences otherwise diluted by zeros. The fixed 12 x 12 grid spans 0-6 on each axis in 0.5 increments, with values >=6 pooled into the last bin. Joint zero mass is 0.890 real versus 0.895 synthetic; marginal u-zero mass is 0.965 versus 0.976, and s-zero mass 0.915 versus 0.906. Tail pooling affects 0.0000 and 0.0000 of pairs, respectively.

TV is descriptive and depends on the sampled cells/genes, raw-count scale, shared zero mass and chosen bins. The pairs are not independent observations and the across-draw range is not a confidence interval. No gene identities or cells are paired between real and synthetic groups.

## Scale check

First draw full-library median: real 7460, synthetic 6592 counts/cell. Sampled nonzero fraction: real 0.110, synthetic 0.105. Sampled mean count per gene-cell: real 0.272, synthetic 0.253. These differences can shift raw covariance, Fano and distance distributions; normalized and gene-z views are sensitivity analyses, not proof of biological calibration.

Failures/limitations: no final metric failures; no independent held-out biological sample, no matched gene identities, no paired-cell comparison, no inferential p-values. See `summary.json` for finite sample counts and full per-draw effects.
