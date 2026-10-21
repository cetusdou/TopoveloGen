# TopoVeloGen

`data_generation_module/` contains the generator and runnable examples.
`scripts/` groups figure and analysis code by experiment; shared helpers are in
`scripts/common/`. `data/results/` contains retained numerical inputs and
outputs. The external pancreas H5AD is not included.

From the repository root, run a generator example with
`python -m data_generation_module.examples.example_generate`.
Run an experiment with `python -m scripts.<group>.<module>`; each figure
pipeline has its own `scripts/figures/<experiment>/` directory.
