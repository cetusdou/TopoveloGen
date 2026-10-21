# Experiments

For current PRE figure numbers, runtime environments, prerequisites, and
output locations, use the [reproduction map](../REPRODUCIBILITY.md).

| Code | Purpose |
| --- | --- |
| `revision_*.py` | Current controlled experiments and numerical checks. Run from the repository root with `python -m experiments.NAME`. |
| `revision_factor_core.py`, `revision_cache.py` | Shared factor-simulation logic and cache provenance guard. |
| `revision_status.py` | Read-only comparison of archived source hashes with this checkout. |
| `common/` | Shared simulation setup, numerical controls, coordinate-aware metrics, and explicit/native projection operators for current runners. |
| `figures/` | Panelized figure pipelines, including earlier figure numbering. |
| `results_*.py`, `figure_4_fast.py` | Retained earlier single-script entrypoints. |

Archive outputs under `../results/` are evidence of their recorded source
states. Use a fresh output directory when a runner supports `--out`; do not
overwrite an archived result merely to regenerate a plot.

The [historical figure recipes](HISTORICAL_FIGURES.md) retain instructions for
the older entrypoints. Their Figure 2--7 labels are not the numbering of the
current manuscript or Supplemental Material.
