"""Figure 6 core: statistical realism (numpy API)."""


import csv
import pickle
import io
import contextlib
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation_module import (  # noqa: E402
    CycleNode,
    GeneSynthesizer,
    GraphEdge,
    GraphNode,
    UniversalGraphSystem,
)
from scripts.figures._common.style import apply_publication_style, save_figure  # noqa: E402
from scripts.topology.results_junction_continuity import init_cycle_rim_cells  # noqa: E402
from scripts.topology.results_topology_primitives import simulate_langevin  # noqa: E402

USE_FIGURE4_PIPELINE = False
VALIDATION_LABEL_PRIMARY = "NB+dropout"
VALIDATION_LABEL_REFERENCE = "Poisson"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 1
N_CELLS = 3000
N_GENES = 1000
N_STEPS = 4500
DT = 0.01
DIFFUSION = 0.006
ALPHA_SCALE = 1.5

OUT_DIR = REPO_ROOT / "data" / "results" / "statistical_realism"
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"

# Figure4Series.ipynb observation-model parameters (fast path: same topology/Langevin)
FIGURE4_OBS = dict(
    dropout_prob=0.25,
    dropout_slope=0.2,
    lib_size_mean=50000.0,
    lib_size_sd=0.2,
    background_noise=0.1,
)
FIGURE4_CELL_SIZE_SIGMA = 0.35
FIGURE4_CAPTURE_SPLICED = 0.20
FIGURE4_CAPTURE_UNSPLICED = 0.15
FIGURE4_NOISE_ALPHA_RANGE = (0.005, 0.05)

APPLY_POSTPROCESS = False
CELL_SIZE_SIGMA = 0.0
CAPTURE_RATE_SPLICED = 1.0
CAPTURE_RATE_UNSPLICED = 1.0
NOISE_ALPHA_RANGE = (0.01, 0.08)

SAVE_EVERY = 10
INIT_RADIUS = 1.0
INIT_NOISE = 0.06
XLIM = (-2.0, 4.0)
YLIM = (-2.5, 2.5)
SYNTH_KW = {
    "edge_strength": 5.0,
    "cycle_strength": 3.0,
    "splicing_rate": 1.0,
    "degradation_rate": 0.5,
    "n_trend_sigmoid": 15,
    "n_trend_impulse": 10,
    "n_cycle_genes": 30,
}


@dataclass
class SynthConfig:
    name: str
    sampling: str = "nb"
    nb_dispersion: float = 1.5
    dropout_prob: float = 0.2
    dropout_slope: float = 0.5
    lib_size_mean: float = 10000.0
    lib_size_sd: float = 0.4
    background_noise: float = 0.02
    simulate_counts: bool = True


MAIN_CONFIG = SynthConfig(name="nb_dropout")
SWEEP_CONFIGS = [
    SynthConfig(name="clean", sampling="poisson", dropout_prob=0.0, simulate_counts=False),
    SynthConfig(name="poisson", sampling="poisson", dropout_prob=0.0, simulate_counts=True),
    SynthConfig(
        name="nb_dropout",
        sampling="nb",
        nb_dispersion=1.5,
        dropout_prob=0.2,
        dropout_slope=0.5,
        simulate_counts=True,
    ),
]


def configure(*, out_dir=None, data_dir=None, fig_dir=None, pipeline: dict | None = None, shared: dict | None = None) -> None:
    global OUT_DIR, DATA_DIR, FIG_DIR
    global SEED, N_CELLS, N_GENES, N_STEPS, DT, DIFFUSION, SAVE_EVERY, ALPHA_SCALE
    global INIT_RADIUS, INIT_NOISE, APPLY_POSTPROCESS, CELL_SIZE_SIGMA
    global CAPTURE_RATE_SPLICED, CAPTURE_RATE_UNSPLICED, NOISE_ALPHA_RANGE
    global MAIN_CONFIG, SWEEP_CONFIGS, SYNTH_KW, XLIM, YLIM
    global VALIDATION_LABEL_PRIMARY, VALIDATION_LABEL_REFERENCE

    if out_dir is not None:
        OUT_DIR = Path(out_dir)
    if data_dir is not None:
        DATA_DIR = Path(data_dir)
    elif out_dir is not None:
        DATA_DIR = OUT_DIR / "data"
    if fig_dir is not None:
        FIG_DIR = Path(fig_dir)
    elif out_dir is not None:
        FIG_DIR = OUT_DIR / "figures"

    if shared:
        VALIDATION_LABEL_PRIMARY = shared.get("validation_label_primary", VALIDATION_LABEL_PRIMARY)
        VALIDATION_LABEL_REFERENCE = shared.get("validation_label_reference", VALIDATION_LABEL_REFERENCE)
        SEED = int(shared.get("seed", SEED))

    if not pipeline:
        return

    SEED = int(pipeline.get("seed", SEED))
    N_CELLS = int(pipeline.get("n_cells", N_CELLS))
    N_GENES = int(pipeline.get("n_genes", N_GENES))
    N_STEPS = int(pipeline.get("n_steps", N_STEPS))
    DT = float(pipeline.get("dt", DT))
    DIFFUSION = float(pipeline.get("diffusion", DIFFUSION))
    SAVE_EVERY = int(pipeline.get("save_every", SAVE_EVERY))
    ALPHA_SCALE = float(pipeline.get("alpha_scale", ALPHA_SCALE))
    INIT_RADIUS = float(pipeline.get("init_radius", INIT_RADIUS))
    INIT_NOISE = float(pipeline.get("init_noise", INIT_NOISE))
    APPLY_POSTPROCESS = bool(pipeline.get("apply_postprocess", APPLY_POSTPROCESS))
    CELL_SIZE_SIGMA = float(pipeline.get("cell_size_sigma", CELL_SIZE_SIGMA))
    CAPTURE_RATE_SPLICED = float(pipeline.get("capture_rate_spliced", CAPTURE_RATE_SPLICED))
    CAPTURE_RATE_UNSPLICED = float(pipeline.get("capture_rate_unspliced", CAPTURE_RATE_UNSPLICED))
    if "noise_alpha_range" in pipeline:
        NOISE_ALPHA_RANGE = tuple(pipeline["noise_alpha_range"])
    if "xlim" in pipeline:
        XLIM = tuple(pipeline["xlim"])
    if "ylim" in pipeline:
        YLIM = tuple(pipeline["ylim"])
    if "main_config" in pipeline:
        MAIN_CONFIG = SynthConfig(**pipeline["main_config"])
    if "sweep_configs" in pipeline:
        SWEEP_CONFIGS = [SynthConfig(**c) for c in pipeline["sweep_configs"]]
    if "synthesizer" in pipeline:
        SYNTH_KW = {**SYNTH_KW, **pipeline["synthesizer"]}

def build_cycle_bifurcation_system() -> UniversalGraphSystem:
    cycle = CycleNode(
        "Cycle",
        pos=[0.0, 0.0],
        radius=1.0,
        potential=-2.5,
        sigma=0.25,
        omega=2.0,
        clockwise=False,
        rotation_axes=(0, 1),
    )
    fate1 = GraphNode(
        "Fate1",
        pos=[3.0, 1.0],
        potential=-3.2,
        sigma=0.30,
        basin_strength=0.75,
        basin_sigma=0.5,
    )
    fate2 = GraphNode(
        "Fate2",
        pos=[3.0, -1.0],
        potential=-3.2,
        sigma=0.30,
        basin_strength=0.75,
        basin_sigma=0.5,
    )
    edges = [GraphEdge(cycle, fate1), GraphEdge(cycle, fate2)]
    return UniversalGraphSystem([cycle, fate1, fate2], edges, fusion_temp=0.01)

def sample_final_cells(system: UniversalGraphSystem, n_cells: int) -> np.ndarray:
    z0 = init_cycle_rim_cells(n_cells, radius=1.0, noise=0.06, seed=SEED)
    _, z_final = simulate_langevin(
        system,
        z0,
        n_steps=N_STEPS,
        dt=DT,
        diffusion=DIFFUSION,
        save_every=10,
        seed=SEED,
    )
    return z_final

def setup_synthesizer(system: UniversalGraphSystem, cfg: SynthConfig) -> GeneSynthesizer:
    synth = GeneSynthesizer(
        system=system,
        n_genes_total=N_GENES,
        edge_strength=float(SYNTH_KW["edge_strength"]),
        cycle_strength=float(SYNTH_KW["cycle_strength"]),
        background_noise=cfg.background_noise,
        lib_size_mean=cfg.lib_size_mean,
        lib_size_sd=cfg.lib_size_sd,
        sampling=cfg.sampling,
        nb_dispersion=cfg.nb_dispersion,
        dropout_prob=cfg.dropout_prob,
        dropout_slope=cfg.dropout_slope,
        splicing_rate=float(SYNTH_KW["splicing_rate"]),
        degradation_rate=float(SYNTH_KW["degradation_rate"]),
        seed=SEED,
    )

    n_sig = int(SYNTH_KW["n_trend_sigmoid"])
    n_imp = int(SYNTH_KW["n_trend_impulse"])
    n_cyc = int(SYNTH_KW["n_cycle_genes"])
    synth.add_trend_gene(edge_idx=0, n_genes=n_sig, pattern="sigmoid", include_cycle_rim=True)
    synth.add_trend_gene(edge_idx=1, n_genes=n_sig, pattern="sigmoid", include_cycle_rim=True)
    synth.add_trend_gene(edge_idx=0, n_genes=n_imp, pattern="impulse", include_cycle_rim=True)
    synth.add_trend_gene(edge_idx=1, n_genes=n_imp, pattern="impulse", include_cycle_rim=True)
    phases = np.linspace(0, 2 * np.pi, n_cyc, endpoint=False)
    synth.add_cycle_gene(node_id="Cycle", n_genes=n_cyc, phase=phases.tolist(), angular_sigma=0.5)
    synth.add_noise_gene(n_genes=N_GENES - synth.current_gene_idx, alpha_range=NOISE_ALPHA_RANGE)

    return synth

def apply_observation_postprocess(s: np.ndarray, u: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Figure4Series steps 2.5–2.6: cell-size scaling + binomial capture downsampling."""
    if not APPLY_POSTPROCESS:
        return s, u

    rng = np.random.default_rng(seed)
    n_cells = s.shape[0]
    mu = -CELL_SIZE_SIGMA**2 / 2
    factors = np.clip(rng.lognormal(mu, CELL_SIZE_SIGMA, n_cells), 0.3, 2.5)
    s = s * factors[:, np.newaxis]
    u = u * factors[:, np.newaxis]

    s = rng.binomial(s.astype(int), CAPTURE_RATE_SPLICED).astype(np.float64)
    u = rng.binomial(u.astype(int), CAPTURE_RATE_UNSPLICED).astype(np.float64)
    return s, u

def generate_counts(
    synth: GeneSynthesizer,
    z: np.ndarray,
    cfg: SynthConfig,
) -> Dict[str, np.ndarray]:
    with contextlib.redirect_stdout(io.StringIO()):
        result = synth.generate_kinetics(
            z,
            dt=DT,
            scale=ALPHA_SCALE,
            simulate_counts=cfg.simulate_counts,
        )
    return result

def to_numpy(x: np.ndarray) -> np.ndarray:
    return x if isinstance(x, np.ndarray) else np.asarray(x)

def compute_cell_stats(S: np.ndarray, U: np.ndarray) -> Dict[str, np.ndarray]:
    total_s = S.sum(axis=1)
    total_u = U.sum(axis=1)
    ratio = total_u / (total_s + 1e-8)
    return {"total_spliced": total_s, "total_unspliced": total_u, "u_s_ratio": ratio}

def compute_gene_stats(X: np.ndarray) -> Dict[str, np.ndarray]:
    mean = X.mean(axis=0)
    var = X.var(axis=0)
    dropout = (X == 0).mean(axis=0)
    return {"mean": mean, "var": var, "dropout": dropout}

def sparsity(X: np.ndarray) -> float:
    return float((X == 0).mean())

def calc_6_metrics(s: np.ndarray, u: np.ndarray) -> Dict[str, np.ndarray]:
    """Six validation metrics from Figure4Series.ipynb (S+U combined counts)."""
    x_total = s + u
    lib_size = x_total.sum(axis=1)
    n_genes = (x_total > 0).sum(axis=1)
    gene_mean = x_total.mean(axis=0)
    gene_var = x_total.var(axis=0)
    mask = gene_mean > 0
    mean_for_drop = gene_mean[mask]
    drop_rate = (x_total[:, mask] == 0).sum(axis=0) / x_total.shape[0]
    u_sum = u.sum(axis=1)
    total_sum = lib_size.copy()
    total_sum[total_sum == 0] = 1
    return {
        "lib_size": lib_size,
        "n_genes": n_genes,
        "gene_mean": gene_mean,
        "gene_var": gene_var,
        "drop_mean": mean_for_drop,
        "drop_rate": drop_rate,
        "u_frac": u_sum / total_sum,
    }

def save_cell_stats_csv(path: Path, z: np.ndarray, cell_stats: Dict[str, np.ndarray]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["cell_idx", "z1", "z2", "total_spliced", "total_unspliced", "u_s_ratio"],
        )
        writer.writeheader()
        for i in range(z.shape[0]):
            writer.writerow(
                {
                    "cell_idx": i,
                    "z1": float(z[i, 0]),
                    "z2": float(z[i, 1]),
                    "total_spliced": float(cell_stats["total_spliced"][i]),
                    "total_unspliced": float(cell_stats["total_unspliced"][i]),
                    "u_s_ratio": float(cell_stats["u_s_ratio"][i]),
                }
            )

def save_gene_stats_csv(path: Path, gene_stats_s: Dict[str, np.ndarray], gene_stats_u: Dict[str, np.ndarray]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "gene_idx",
                "mean_spliced",
                "var_spliced",
                "dropout_spliced",
                "mean_unspliced",
                "var_unspliced",
                "dropout_unspliced",
            ],
        )
        writer.writeheader()
        for g in range(len(gene_stats_s["mean"])):
            writer.writerow(
                {
                    "gene_idx": g,
                    "mean_spliced": float(gene_stats_s["mean"][g]),
                    "var_spliced": float(gene_stats_s["var"][g]),
                    "dropout_spliced": float(gene_stats_s["dropout"][g]),
                    "mean_unspliced": float(gene_stats_u["mean"][g]),
                    "var_unspliced": float(gene_stats_u["var"][g]),
                    "dropout_unspliced": float(gene_stats_u["dropout"][g]),
                }
            )

def save_count_summary_csv(path: Path, rows: List[Dict]) -> None:
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def build_plot_bundle(
    system: UniversalGraphSystem,
    z: np.ndarray,
    s: np.ndarray,
    u: np.ndarray,
    s_poisson: np.ndarray,
    u_poisson: np.ndarray,
    sweep_sparsity: Dict[str, Tuple[float, float]],
) -> Dict:
    return {
        "system": system,
        "z": z,
        "s": s,
        "u": u,
        "s_poisson": s_poisson,
        "u_poisson": u_poisson,
        "sweep_sparsity": sweep_sparsity,
        "cell_stats": compute_cell_stats(s, u),
        "gene_s": compute_gene_stats(s),
        "cell_poisson": compute_cell_stats(s_poisson, u_poisson),
        "stats_primary": calc_6_metrics(s, u),
        "stats_reference": calc_6_metrics(s_poisson, u_poisson),
        "label_primary": VALIDATION_LABEL_PRIMARY,
        "label_reference": VALIDATION_LABEL_REFERENCE,
    }

def load_plot_bundle() -> Dict:
    count_path = DATA_DIR / "count_matrices.pt"
    if not count_path.exists():
        raise FileNotFoundError(
            f"Missing cached count data: {count_path}\n"
            "Run: python scripts/figures/fig06_statistical_realism/pipeline/run_pipeline.py"
        )
    data = pickle.load(open(str(count_path), "rb"))
    z = to_numpy(data["z"])
    S = to_numpy(data["spliced"]).astype(np.float64)
    U = to_numpy(data["unspliced"]).astype(np.float64)
    S_poisson, U_poisson = load_poisson_from_cache(data)
    sweep_sparsity = load_sweep_sparsity()
    system = build_cycle_bifurcation_system()
    return build_plot_bundle(system, z, S, U, S_poisson, U_poisson, sweep_sparsity)

def load_poisson_from_cache(data: dict) -> Tuple[np.ndarray, np.ndarray]:
    if "spliced_poisson" in data:
        return (
            to_numpy(data["spliced_poisson"]).astype(np.float64),
            to_numpy(data["unspliced_poisson"]).astype(np.float64),
        )
    print("  Poisson baseline not cached; regenerating from latent positions...")
    z = to_numpy(data["z"])
    system = build_cycle_bifurcation_system()
    poisson_cfg = next(c for c in SWEEP_CONFIGS if c.name == "poisson")
    synth = setup_synthesizer(system, poisson_cfg)
    res = generate_counts(synth, np.asarray(z, dtype=np.float32), poisson_cfg)
    s_poisson = to_numpy(res["spliced"]).astype(np.float64)
    u_poisson = to_numpy(res["unspliced"]).astype(np.float64)
    return apply_observation_postprocess(s_poisson, u_poisson, SEED + 42)

def load_sweep_sparsity() -> Dict[str, Tuple[float, float]]:
    summary_path = DATA_DIR / "count_statistics.csv"
    if not summary_path.exists():
        return {}
    sweep: Dict[str, Tuple[float, float]] = {}
    with summary_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["setting"] == "main_nb_dropout":
                continue
            sweep[row["setting"]] = (float(row["sparsity_spliced"]), float(row["sparsity_unspliced"]))
    return sweep

def run_pipeline() -> Dict:
    """Generate count caches + CSVs. Returns plot bundle."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Building topology & sampling cells...")
    system = build_cycle_bifurcation_system()
    z = sample_final_cells(system, N_CELLS)
    z_np = to_numpy(z)

    print(f"Generating main count matrix ({MAIN_CONFIG.name}, {N_GENES} genes)...")
    synth_main = setup_synthesizer(system, MAIN_CONFIG)
    result_main = generate_counts(synth_main, z, MAIN_CONFIG)
    S = to_numpy(result_main["spliced"]).astype(np.float64)
    U = to_numpy(result_main["unspliced"]).astype(np.float64)
    S, U = apply_observation_postprocess(S, U, SEED)

    cell_stats = compute_cell_stats(S, U)
    gene_s = compute_gene_stats(S)
    gene_u = compute_gene_stats(U)
    save_cell_stats_csv(DATA_DIR / "cell_statistics.csv", z_np, cell_stats)
    save_gene_stats_csv(DATA_DIR / "gene_statistics.csv", gene_s, gene_u)

    summary_rows = [
        {
            "setting": "main_nb_dropout",
            "n_cells": N_CELLS,
            "n_genes": N_GENES,
            "sparsity_spliced": sparsity(S),
            "sparsity_unspliced": sparsity(U),
            "median_total_spliced": float(np.median(cell_stats["total_spliced"])),
            "median_total_unspliced": float(np.median(cell_stats["total_unspliced"])),
            "median_u_s_ratio": float(np.median(cell_stats["u_s_ratio"])),
        }
    ]

    print("Noise-setting sweep...")
    sweep_sparsity: Dict[str, Tuple[float, float]] = {}
    S_poisson = U_poisson = None
    for cfg in SWEEP_CONFIGS:
        synth = setup_synthesizer(system, cfg)
        res = generate_counts(synth, z, cfg)
        s_mat = to_numpy(res["spliced"]).astype(np.float64)
        u_mat = to_numpy(res["unspliced"]).astype(np.float64)
        s_mat, u_mat = apply_observation_postprocess(s_mat, u_mat, SEED + hash(cfg.name) % 10000)
        if cfg.name == "poisson":
            S_poisson, U_poisson = s_mat, u_mat
        sp_s, sp_u = sparsity(s_mat), sparsity(u_mat)
        sweep_sparsity[cfg.name] = (sp_s, sp_u)
        summary_rows.append(
            {
                "setting": cfg.name,
                "n_cells": N_CELLS,
                "n_genes": N_GENES,
                "sparsity_spliced": sp_s,
                "sparsity_unspliced": sp_u,
                "median_total_spliced": float(np.median(s_mat.sum(axis=1))),
                "median_total_unspliced": float(np.median(u_mat.sum(axis=1))),
                "median_u_s_ratio": float(np.median(u_mat.sum(axis=1) / (s_mat.sum(axis=1) + 1e-8))),
            }
        )
        print(f"  {cfg.name}: sparsity S={sp_s:.3f}, U={sp_u:.3f}")

    assert S_poisson is not None and U_poisson is not None
    save_count_summary_csv(DATA_DIR / "count_statistics.csv", summary_rows)
    pickle.dump(
        {
            "z": np.asarray(z),
            "spliced": np.asarray(S),
            "unspliced": np.asarray(U),
            "spliced_poisson": np.asarray(S_poisson),
            "unspliced_poisson": np.asarray(U_poisson),
        },
        open(str(DATA_DIR / "count_matrices.pt"), "wb"),
    )
    return build_plot_bundle(system, z_np, S, U, S_poisson, U_poisson, sweep_sparsity)

