# Code Modularization Design

## Objective and baseline

Reorganize the code repository into three recognizable parts: generator,
experiments, and data. The baseline is commit `89b59fa`, preserved by the
`pre-modularization-2026-10-21` tag. Work proceeds on
`codex/modularize-code`; the paper repository is outside this change.

The goal is maintainable modules and clear ownership, not preserving old
source hashes, byte-identical outputs, or historical path names. Existing
scientific definitions and current manuscript analyses should nevertheless
remain functionally comparable. Any intentional numerical change must be
identified rather than hidden in a file move.

## 1. Generator

`data_generation_module` remains the public Python package. Its internal
modules should be grouped by responsibility: graph geometry and drift,
transcription fields and RNA kinetics, count observation, and topology
presets. `GraphSimulator`, `GeneSynthesizer`, and the graph node types are
currently the important public objects. Their imports through
`data_generation_module.__init__` remain valid while implementation moves
behind those imports. Visualization helpers should not be a dependency of
simulation modules.

The generator owns model equations and sampling. It must not import from
`experiments` or assume a paper-figure output directory. Existing callers
may use a compatibility import during migration, but new code should import
the module that owns the behavior.

## 2. Experiments

Current `revision_*.py` workflows remain distinguishable from older figure
scripts. Shared simulation setup, coordinate transformations, metrics,
projection operators, provenance/cache handling, and plotting should be
extracted only where multiple current workflows use the same behavior.
An experiment runner owns its configuration, orchestration, output schema,
and command-line interface; it should not import a peer runner solely to
reuse one helper function.

Group current runners by purpose: numerical/model checks, velocity-method
benchmarking, projection analysis, and reference-guided/data statistics.
Keep executable `python -m ...` entry points or provide thin forwarding
wrappers during migration. Historical `results_*.py`, older monoliths, and
panelized figure packages form a clearly labeled legacy area; they are not
silently rewritten as if they were the current benchmark.

## 3. Data

Separate externally supplied inputs, experiment configuration, generated
datasets, and final summaries/figures. A small path/configuration layer
resolves repository-relative defaults and supports explicit overrides for
external inputs and output roots. It must report missing inputs clearly and
must not download or synthesize an external reference implicitly.

The pancreas `.h5ad` file is an external input and remains untracked. Large
existing result archives may be migrated after their readers and writers use
the new path layer. Migration should use a documented old-to-new path map and
should never discard the only copy of a dataset. The former hash-based
archive identity is not a migration constraint.

## Dependency direction and migration

The dependency direction is `experiment runner -> shared experiment helpers
-> generator`; data paths/configuration are consumed by runners and helpers.
Generator code must not depend on experiments, figure styles, or the paper
repository. Data loaders return validated arrays/objects rather than
embedding figure-specific analysis.

Implement in three reviewable increments, each with its own commit:

1. Split and stabilize generator internals, retaining public imports.
2. Extract shared experiment logic and regroup current runners; preserve
   executable entry points while updating internal imports.
3. Centralize data paths and loaders, migrate selected active datasets and
   outputs, update documentation, and label remaining archives explicitly.

Each increment runs focused unit or smoke checks and at least one relevant
small end-to-end workflow. Compare key shapes, labels, and numerical metrics
against the baseline where feasible. Full expensive grids are not required
for every increment. Do not edit manuscript TeX or rewrite archived results
merely to make checks pass.

## Acceptance

- The three areas and their responsibilities are discoverable from the root
  README and package documentation.
- The generator public API and current experiment commands have working
  import/CLI checks; old paths are either migrated or documented as legacy.
- Current runners no longer import peer runners for reusable model/metric
  helpers.
- External input and output locations are centrally resolved, with clear
  failures for unavailable inputs.
- The code diff, focused checks, and any numerical differences are reviewed
  before each commit; the paper repository remains untouched.
