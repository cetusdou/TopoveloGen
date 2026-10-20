# TopoVeloGen code and experiments

This is the code repository nested beside the PRE manuscript. It has its own
Git history. The parent manuscript repository does not track this directory.

## Layout

| Directory | Responsibility |
| --- | --- |
| `data_generation_module/` | Graph scaffold, latent drift, transcription programs, RNA/count generation, and public examples. |
| `experiments/figures/` | Panelized figure pipelines with JSON settings and per-figure instructions. Their historical figure numbers are not necessarily the current manuscript numbers. |
| `experiments/revision_*.py` | Reviewer-revision numerical experiments and analysis entrypoints. Shared factor simulation is in `revision_factor_core.py`; cache checking is in `revision_cache.py`. |
| `results/` | Archived inputs, per-seed outputs, summaries, figures, and provenance. Treat existing archives as immutable. |
| `data/` | External biological inputs. The pancreas H5AD is intentionally not in Git. |

The current manuscript-to-code map, execution order, and environment details
are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). For the generator API, see
[data_generation_module/README.md](data_generation_module/README.md).

## Read-only first check (Windows PowerShell)

From this repository root, inspect recorded source hashes without installing
scientific dependencies or rewriting result archives:

```powershell
py -3.10 -m experiments.revision_status
```

The current checkout does not contain the former Python requirements files.
Consequently, it does not provide a complete fresh-install recipe. Existing
local environments and recorded package versions are described in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md); the three-method benchmark needs a
separate legacy environment. Do not mix those package stacks.

After provisioning the appropriate environment, rerun factor experiments in a
**new** output directory:

```powershell
.\.venv-revision\Scripts\python.exe -m experiments.revision_factor_regions --out results/new_factor_run
```

The factor, three-factor interaction, and cell/gene runners compare the
settings and source hashes with `config.json` before reusing cached datasets.
They refuse a mismatched archive rather than mixing old and new results.

Some full grids take substantial time and disk space; check their design and
prerequisites first.
