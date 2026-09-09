# Experiments

All result figures use the shared publication style in `styles/nature_science.mplstyle`
(via `experiments/figure_style.py`). Each script supports `--plot-only` to replot from
cached data without rerunning simulations.

## Figure 2 — topology primitives (`results_topology_primitives.py`)

Landscape gallery demonstrating controllable latent manifolds from user-defined graph primitives:
linear, bifurcation, cycle, cycle–linear hybrid, skeleton S-curve, and cycle–bifurcation hybrid.
Each panel uses the same visual language (landscape + prescribed skeleton + trajectories/cells).
Distance-to-manifold quantification is saved to CSV and plotted as Supplementary Fig. S1 (not in the main composite).

**Run** (from repository root; requires PyTorch + matplotlib):

```bash
/home/doujy/miniconda3/envs/torch/bin/python experiments/results_topology_primitives.py
python experiments/results_topology_primitives.py --plot-only
```

**Outputs**

| Path | Description |
|------|-------------|
| `results/topology_primitives/data/topology_metrics.csv` | Distance-to-manifold and flow-alignment metrics |
| `results/topology_primitives/data/*_traj.npz` | Saved trajectories per topology |
| `results/topology_primitives/figures/panel_*.png` | Individual panels A–F |
| `results/topology_primitives/figures/figure2_topology_primitives.png` | Combined 2×3 landscape gallery |
| `results/topology_primitives/figures/supplementary_s1_topology_quantification.png` | Distance-to-manifold bar chart |

Default hyperparameters are defined at the top of the script (`N_CELLS`, `N_STEPS`, `DIFFUSION`, etc.).

Drift **streamplot** uses a manifold support mask (ρ_max > `SUPPORT_CUT`; drift zeroed outside). Panels are square (`set_box_aspect(1)` + equal x/y span). Colormap: `RdBu`.

## Figure 3 — junction continuity (`results_junction_continuity.py`)

Tests soft fusion and smart rim docking at cycle–linear, bifurcation, and cycle–bifurcation junctions.

```bash
/home/doujy/miniconda3/envs/torch/bin/python experiments/results_junction_continuity.py
python experiments/results_junction_continuity.py --plot-only
```

## Figure 4 — alpha fields (`results_alpha_fields.py`)

Maps latent positions to transcription rates α(z) via `GeneSynthesizer` on cycle-bifurcation topology (no ODE / u,s).

Gene types shown: branch sigmoid (B/C), **junction exit impulse** at cycle-rim docks (D), cycle phase (E/F), plus noise genes in the matrix (not panelled). Sticky lineage is omitted here (needs a downstream-edge chain topology).

```bash
/home/doujy/miniconda3/envs/torch/bin/python experiments/results_alpha_fields.py
python experiments/results_alpha_fields.py --plot-only
```

Outputs: `results/alpha_fields/data/{latent_positions,alpha_matrix}.pt`, `gene_metadata.csv`, `alpha_field_metrics.csv`, and `figures/figure4_alpha_fields.{png,pdf}`.

## Figure 5 — kinetic validation (`results_kinetic_validation.py`)

Integrates unspliced/spliced ODEs along a deterministic linear trajectory; validates splicing delay, u–s hysteresis, γ sensitivity, and ground-truth velocity consistency.

```bash
/home/doujy/miniconda3/envs/torch/bin/python experiments/results_kinetic_validation.py
python experiments/results_kinetic_validation.py --plot-only
```

Outputs: `results/kinetic_validation/data/*.csv`, `kinetic_tensors.pt`, and `figures/figure5_kinetic_validation.{png,pdf}`.

## Figure 6 — count statistical realism (`results_statistical_realism.py`)

Converts kinetic outputs to Poisson/NB count matrices with library-size sampling and dropout; validates library-size, mean–variance, dropout, and sparsity.

```bash
/home/doujy/miniconda3/envs/torch/bin/python experiments/results_statistical_realism.py
python experiments/results_statistical_realism.py --plot-only
```

Outputs: `results/statistical_realism/data/{cell,gene,count}_statistics.csv`, `count_matrices.pt`, and `figures/figure6_statistical_realism.{png,pdf}`.

## Figure 7 — embedding masked errors (`results_embedding_masked_errors.py`)

Benchmarks scVelo (stochastic mode) on synthetic count data with known high-dimensional ground-truth velocity. Compares low-D embedding agreement vs high-D kinetic accuracy; defines mismatch cells as low-D correct but high-D wrong.

Requires scVelo, scanpy, anndata. Set `NUMBA_CACHE_DIR` if numba caching fails:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache /home/doujy/miniconda3/envs/torch/bin/python experiments/results_embedding_masked_errors.py
python experiments/results_embedding_masked_errors.py --plot-only   # replot from cache
```

Figures use `styles/nature_science.mplstyle` (Nature/Science publication style).

Outputs: `results/embedding_masked_errors/data/{adata_velocity_benchmark.h5ad,velocity_accuracy_metrics.csv,mismatch_by_region.csv,mismatch_by_topology.csv}` and `figures/figure7_lowD_highD_mismatch.{png,pdf}`.
