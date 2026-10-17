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

## Quick start (Windows PowerShell)

Run commands from this repository root with Python 3.10:

```powershell
py -3.10 -m venv .venv-revision
.\.venv-revision\Scripts\python.exe -m pip install -r requirements-revision.txt
.\.venv-revision\Scripts\python.exe -m experiments.revision_coordinate_projection --help
```

The three-method benchmark uses a separate legacy dependency stack documented
in [requirements-methods.txt](requirements-methods.txt) and
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). Installing those versions into the
revision environment would change the numerical environment.
An exact-version snapshot of the current revision environment is in
[requirements-revision-lock.txt](requirements-revision-lock.txt); it does not
include wheel hashes.

To rerun factor experiments after editing code, use a **new** output directory:

```powershell
.\.venv-revision\Scripts\python.exe -m experiments.revision_factor_regions --out results/new_factor_run
```

The factor, three-factor interaction, and cell/gene runners compare the
settings and source hashes with `config.json` before reusing cached datasets.
They refuse a mismatched archive rather than mixing old and new results.

No experiment is run during installation. Some full grids take substantial
time and disk space; check their design and prerequisites first.

To compare archived source hashes with the current checkout without running
any simulation:

```powershell
.\.venv-revision\Scripts\python.exe -m experiments.revision_status
```
