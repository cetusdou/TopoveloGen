# Reproduction map for the current manuscript

## Repository boundary

This code repository is independent of the parent paper repository. Manuscript
assets live at `../figure/`; experiment outputs live at `results/`. The paper
does not automatically regenerate figures when a script runs. Before copying a
new plot into the paper, record the script, command, environment, output
directory, input hash, and changed parameters.

The historical `experiments/figures/fig*/` packages are already panelized.
Their numbered README files describe an earlier figure organization. Use the
table below, not those historical numbers, for the **current** paper. The
`results_*.py` and `figure_4_fast.py` scripts remain compatibility entrypoints;
do not delete them merely because a panelized replacement exists.

## Environments and inputs

| Task | Runtime | Notes |
| --- | --- | --- |
| Generator and most revision controls | Python 3.10, `requirements-revision.txt` | Observed environment uses scVelo 0.3.3. The requirements file pins direct dependencies, not every transitive wheel. |
| scVelo / UniTVelo / DeepVelo operational comparison | Separate Python 3.9, `requirements-methods.txt` | Observed environment uses scVelo 0.2.5, UniTVelo 0.2.5.2, DeepVelo 0.2.4, PyTorch 1.12.1, DGL 0.6.1. The recorded PyTorch wheel is CPython 3.9 for Windows; old PyTorch/DGL wheels may require platform-specific installation. |
| scMultiSim reference generation | R with scMultiSim 1.2.0 | `experiments/revision_scmultisim_comparison.R` uses `.r-methods` if present. Record the R and package versions separately. |
| Pancreas reference | `data/Pancreas/endocrinogenesis_day15.h5ad` | External input, deliberately ignored by Git. Verify the SHA-256 in the corresponding experiment manifest before claiming an exact replay. |

`requirements-revision-lock.txt` records the full installed Python package
version set for the revision environment. The methods environment used local
PyTorch/UniTVelo/DeepVelo wheels, so `requirements-methods.txt` records their
versions without embedding machine-specific `file:///` paths. Neither file
locks the R environment or the external pancreas input.

Do not install both Python stacks in one virtual environment. The existing
`.venv-revision`, `.venv-methods`, `.r-methods`, and local wheel directories are
machine-specific and ignored by Git. A fresh installation may need historical
wheels or repository-specific package sources; the listed versions describe the
working environments, not a promise that all packages remain downloadable.

## Figure-to-source map

Run Python commands from this repository root as `python -m experiments.NAME`.
The right column names the local output directory; paper PDFs are copied from
there after review. A PDF alone is not the complete reproducibility record.

| Current paper figure | Primary code entrypoint | Local output / prerequisite |
| --- | --- | --- |
| Fig. 1 | Manuscript schematic | `../figure/Figure1.png`; no simulation run. |
| Fig. 2 | `experiments/figures/fig02_topology_primitives/pipeline/run_pipeline.py` | `results/topology_primitives/`; historical panelized pipeline. |
| Fig. 3 | `experiments/reproduce_figure3_series.py` and `experiments/compose_figure3_series_big.py` | `results/figure3_series/`; panel-level settings and source history should be checked before claiming pixel-identical reconstruction. |
| Fig. 4 | `experiments/figure4_series_pipeline.py` / `experiments/figures/fig4series_pancreas_stats/` | `results/statistical_realism_figure4/`; requires the external pancreas input. |
| Fig. 5F--H / response R7 | `experiments/revision_three_method_benchmark.py` | `results/revision_three_method_benchmark/`; staged `prepare`, `fit`, `summarize` workflow in the separate methods environment. |
| Fig. 6F--H / response R5 | `experiments/revision_projection_closure.py` | `results/revision_projection_closure/`; reads the archived coordinate-control datasets. |
| S1 / R6 | `experiments/revision_coordinate_projection.py` | `results/revision_coordinate_projection/`; also supplies inputs for several projection controls. |
| S2 / R3 | `experiments/revision_numerical_kinetics.py` | `results/revision_numerical_kinetics/`; latent/RNA numerical summary. |
| S3 / R16 | `experiments/revision_numerical_kinetics.py` | `results/revision_numerical_kinetics/`; harmonic phase response. |
| S4 / R2 | `experiments/revision_figure8_junction.py` | `results/revision_figure8_junction/`; crossing audit. |
| S5 / R13 | `experiments/revision_coordinate_projection.py` and `experiments/response_sensitivity_panels.py` | `results/revision_coordinate_projection/`; threshold/norm-floor summaries. |
| S6 / R11 and S7 / R12 | `experiments/revision_factor_regions.py` | `results/revision_factor_regions/`; shared simulation core in `revision_factor_core.py`. |
| S8 / R8 | `experiments/revision_interaction_scan.py` | `results/revision_interaction_scan/`; diffusion/kinetics/dropout crossing. |
| S9 / R9 | `experiments/revision_cell_gene_factor_closure.py` | `results/revision_cell_gene_factor_closure/`; cell/gene and diffusion/dropout crossings. |
| S10 / R4 | `experiments/revision_topology_numerics_closure.py` | `results/revision_topology_numerics_closure/`; start-group junction sensitivity. |
| S11 / R14 | `experiments/revision_main_metric_timestep.py` | `results/revision_main_metric_timestep/`; time-step sensitivity of velocity scores. |
| S12 / R10 | `experiments/revision_scmultisim_comparison.R`, then `experiments/revision_cross_simulator.py` | `results/revision_scmultisim_comparison/`; Python also reads the three-method archive. |
| S13 / R15 | `experiments/revision_branch_convergence.py` | `results/revision_branch_convergence/`; branch-label refinement. |
| S14 / R1 | `experiments/revision_escape_scan.py` | `results/revision_escape_scan/`; finite-horizon first-exit scan. |
| S15 / R17 | `experiments/revision_pancreas_higher_order.py` | `results/revision_pancreas_higher_order/`; requires the pancreas reference and synthetic count input. |
| S16--S17 | Enlarged main-text Figs. 5A--E and 6A--E | Archival displays, not outputs of the controlled Fig. 5F--H and 6F--H runners. |

For main Figs. 2--4, the figure-series pipelines and paper assets are related
but not asserted to be the same byte-for-byte run. The manuscript parameter
tables and archived manifests are the authority for the reported experiments.
Do not silently replace an archived asset with a new pipeline output.

## Running and preserving results

1. Record `git rev-parse HEAD` and `git status --short` in **this** repository.
2. Use the matching Python or R environment above and verify external input
   hashes. Run prerequisite stages before downstream analyses.
3. For changed factor experiments, choose a new directory with `--out`, e.g.
   `python -m experiments.revision_interaction_scan --out results/interaction_new`.
   The factor, interaction, and cell/gene scripts reject a cached archive whose
   `config.json` differs from the current settings or source hashes.
4. Other revision scripts still have fixed output locations. Run them in a
   clean checkout or copy the current `results/` archive elsewhere first;
   do not assume every older script protects existing files from overwrite.
5. Retain per-seed files, `config.json` or provenance manifests, and the
   generated summary alongside any figure exported to `../figure/`.

`python -m experiments.revision_status` audits recorded source hashes against
the current checkout without changing results. `DIFFERENT` is expected after
code edits and means the archived output belongs to an earlier source state;
it is not evidence that its stored numbers are wrong. `--strict` exits nonzero
if a recorded source differs or an archive has no source hashes at all. A zero
exit would cover only **recorded** source files, not unrecorded dependencies,
package wheels, external data, or numerical identity across platforms.

No figure comparison here should treat multiple cells from one dataset as
independent simulation replicates. The manuscript parameter tables give the
replication unit for each experiment.
