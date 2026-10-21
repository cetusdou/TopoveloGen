# Code Modularization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Separate generator, current experiment helpers, and data path ownership without changing the paper repository.

**Architecture:** Keep `data_generation_module` and the current `revision_*.py` commands as public entry points. Move reusable implementation into owner modules, and introduce one path layer for external inputs and generated outputs. Leave historical figure scripts identifiable and avoid an indiscriminate 220-file rename.

**Tech Stack:** Python 3.10+ (existing revision environment), NumPy, SciPy, matplotlib, `unittest` from the standard library.

**Spec:** `docs/superpowers/specs/2026-10-21-code-modularization-design.md`

## Global Constraints

- Baseline: tag `pre-modularization-2026-10-21` at `89b59fa`; work on `codex/modularize-code`.
- Do not edit paper TeX, paper figures, or manuscript results.
- Old source hashes and exact byte identity are not migration constraints; scientific changes must be called out.
- Do not overwrite the only copy of an external input or an archived dataset.
- Commit each completed module of work separately.

---

### Task 1: Generator observation sampling

**Files:** Create `data_generation_module/observation.py`, `tests/__init__.py`, and `tests/test_observation.py`; modify `data_generation_module/synthesizer_gene.py`; update `data_generation_module/README.md`.

**Interfaces:** `draw_capture_factor(expr_rate, lib_size_mean, lib_size_sd)` returns an `(n_cells, 1)` NumPy array. `sample_counts(expr_rate, capture_factor, capture_scale, sampling, nb_dispersion, dropout_prob, dropout_slope)` returns sampled counts. Existing `GeneSynthesizer._draw_capture_factor` and `_sample_counts` remain delegation wrappers.

- [x] Write `unittest` coverage for capture shape and deterministic Poisson/negative-binomial/dropout output under fixed NumPy seeds, comparing against the baseline methods.
- [x] Run `python -m unittest tests.test_observation -v`; confirm failure before implementation.
- [x] Move the exact sampling operations from `GeneSynthesizer` into `observation.py`, preserving random draw order, dtypes, and fallback behavior. Keep wrappers with unchanged signatures.
- [x] Run the focused tests, `python -m compileall -q data_generation_module`, and a small synthesizer smoke run. Inspect the diff for numerical changes; commit only this task.

The wrapper calls the new function with its existing instance parameters:

```python
return sample_counts(expr_rate, capture_factor, capture_scale,
                     self.sampling, self.nb_dispersion,
                     self.dropout_prob, self.dropout_slope)
```

### Task 2: Experiment metrics and projection helpers

**Files:** Create `experiments/common/__init__.py`, `experiments/common/velocity_metrics.py`, `experiments/common/projection.py`, and `tests/test_velocity_metrics.py`; modify `experiments/revision_coordinate_projection.py`, `experiments/revision_projection_closure.py`, `experiments/revision_cell_gene_factor_closure.py`, `experiments/revision_main_metric_timestep.py`, and `experiments/revision_factor_core.py` only where those helpers are imported.

**Interfaces:** `cosine(a, b, relative_floor=1e-6)` and `pushforward(s, v, target=10000.0)` retain their existing array semantics. `positive_cosine_neighbor_projection(v, x, y, indices)` implements the current `kernel_project` operation. Runner modules may keep thin aliases temporarily for external callers.

- [x] Add fixed-array tests for finite/zero vector masks, normalized/log pushforward, and neighbor projection; run them and confirm missing-module failure.
- [x] Extract the existing formulas without changing denominators, floors, or validity masks. Replace peer-runner imports of these helpers with imports from `experiments.common`.
- [x] Run focused tests, compileall, and `--help` on the affected runnable modules. Inspect the diff and commit this task.

The first extraction preserves the existing cosine validity rule:

```python
valid = np.isfinite(a).all(1) & np.isfinite(b).all(1) & (na > fa) & (nb > fb)
```

### Task 3: Data paths and external input validation

**Files:** Create `experiments/data_paths.py` and `tests/test_data_paths.py`; modify `experiments/revision_pancreas_higher_order.py` and `experiments/figure4_series_pipeline.py` only for shared default-path resolution; update `README.md` and `REPRODUCIBILITY.md`.

**Interfaces:** `repository_root()` returns the code repo root; `pancreas_reference(override=None, require=True)` resolves an explicit path or the default `data/Pancreas/endocrinogenesis_day15.h5ad` and raises `FileNotFoundError` with the path when required input is absent; `result_dir(name, root=None)` returns an output path without creating it.

- [x] Add `unittest` coverage for default/override path resolution and missing-input failure; confirm failure before implementation.
- [x] Introduce the path functions. Replace only duplicated active pancreas defaults and leave historical scripts unchanged. Do not move the 52 MB untracked reference file or any result archive.
- [x] Run focused tests, compileall, and the active entrypoint `--help` checks. Update the root documentation to distinguish external `data/`, generated `results/`, and historical archives; commit this task.

The default input comes from the repository, not a machine-specific path:

```python
default = repository_root() / "data" / "Pancreas" / "endocrinogenesis_day15.h5ad"
```

### Task 4: Integration and final review

**Files:** Update `experiments/README.md`, `data_generation_module/README.md`, and `REPRODUCIBILITY.md` only if Task 1-3 documentation leaves gaps; no paper files.

- [x] Run all new `unittest` modules, compileall for generator/experiments, and import checks for public generator classes and shared helpers.
- [x] Run a small current simulation in memory without writing an archived result directory.
- [x] Search for stale imports of extracted helpers and broken documentation links. Review `git diff` from baseline for accidental scientific changes.
- [x] Commit any integration-only corrections separately and report exact verification results and remaining unmigrated historical scripts.

## Completion record (2026-10-21)

Generator count sampling, shared experiment operations, and active data paths
were extracted in separate commits. The 16 focused unittest checks and
compileall passed. A ten-cell coordinate simulation and the local Figure 4
pancreas reference load succeeded. Full factor grids and method fitting were
not rerun; historical figure scripts and existing result archives were not
moved. The Figure 4 loader now fails explicitly on a missing input instead of
substituting the scVelo example dataset.
