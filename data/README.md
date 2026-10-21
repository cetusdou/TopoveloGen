# External inputs

`data/` contains externally supplied biological inputs, not generated
simulation results. The Figure 4 pancreas reference is expected at
`data/Pancreas/endocrinogenesis_day15.h5ad`; this large file is intentionally
untracked. Current experiments resolve this default through
`experiments.data_paths.pancreas_reference` and fail with an explicit path if
the file is missing. Pass an explicit path to the loader to use another input.

Generated datasets, summaries, and figures belong under `results/`. Existing
archives have not been moved as part of this module split.
