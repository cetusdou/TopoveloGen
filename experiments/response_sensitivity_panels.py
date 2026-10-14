"""Render existing seed-level sensitivity outputs; no simulations are rerun."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/revision_coordinate_projection/sensitivity.csv"
OUT = ROOT.parent / "figure/RevisionSensitivity.pdf"


def main():
    df = pd.read_csv(SOURCE)
    topologies = ["cycle", "bifurcation", "cycle_bifurcation"]
    fig, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
    styles = {"gene": ("#555555", "Gene"), "pca_linear": ("#2878b5", "PCA"),
              "umap_kernel": ("#188977", "UMAP kernel")}
    for col, topology in enumerate(topologies):
        sub = df[df.topology == topology]
        ax = axes[0, col]
        selected = sub[sub.projection == "umap_kernel"]
        assert selected.groupby(["high", "low"]).size().eq(5).all()
        grid = selected.groupby(["high", "low"]).fraction.mean().unstack("low").sort_index()
        ax.imshow(100 * grid.to_numpy(), vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
        for i in range(3):
            for j in range(3):
                val = grid.iloc[i, j] * 100
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        color="white" if val > 55 else "black", fontsize=10)
        ax.set_xticks(range(3), [f"{x:g}" for x in grid.columns])
        ax.set_yticks(range(3), [f"{x:g}" for x in grid.index])
        ax.set_xlabel("Gene cosine cutoff (below)")
        ax.set_ylabel("Projected cosine cutoff (above)")
        ax.set_title(topology.replace("_", "-"), fontsize=12)
        for name, (color, label) in styles.items():
            raw = sub[sub.projection == "norm_floor_" + name]
            for row, field, multiplier in [(1, "fraction", 100), (2, "mean_delta", 1)]:
                grouped = raw.groupby("high")[field].agg(["mean", "std", "count"])
                assert grouped["count"].eq(5).all()
                err = t.ppf(.975, 4) * grouped["std"] / np.sqrt(5)
                axes[row, col].errorbar(grouped.index, grouped["mean"] * multiplier,
                    yerr=err * multiplier, marker="o", color=color, label=label, capsize=3)
                axes[row, col].set_xscale("log")
                axes[row, col].set_xlabel("Relative norm floor")
        axes[1, col].set_ylabel("Pairwise-valid cells (%)")
        axes[2, col].set_ylabel("Mean paired cosine change")
        axes[2, col].axhline(0, color="0.7", lw=.7)
        axes[1, col].legend(fontsize=8, loc="best")
        for row in (1, 2):
            axes[row, col].spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT)
    plt.close(fig)
    meta = {"source": str(SOURCE.relative_to(ROOT)),
            "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "output_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
            "estimand": "Top: mean threshold fraction; middle: valid fraction; bottom: paired cosine change. Five independent dataset seeds; 95% t intervals on line plots.",
            "new_simulation": False}
    record = ROOT.parent / "build/revision_archive/sensitivity_figure_provenance.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
