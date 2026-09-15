"""Shared publication matplotlib style for TopoVeloGen result figures."""


from pathlib import Path
from typing import Iterable, Union

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
STYLE_PATH = REPO_ROOT / "styles" / "nature_science.mplstyle"

# Double-column 2×3 composite figure (square panels: 3 wide × 2 tall)
FIGURE_2X3_SIZE = (9.0, 6.0)
FIGURE_SINGLE_SIZE = (3.0, 3.0)


def apply_publication_style() -> None:
    if not STYLE_PATH.exists():
        raise FileNotFoundError(f"Missing matplotlib style: {STYLE_PATH}")
    plt.style.use(str(STYLE_PATH))
    # Keep text editable in vector outputs instead of converting glyphs to paths.
    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "pdf.compression": 9,
        }
    )


def save_figure(
    fig: plt.Figure,
    out_path: Union[Path, str],
    formats: Iterable[str] = ("png", "pdf"),
) -> None:
    """Save *fig* to every format in *formats*.

    Text is always kept editable in vector outputs (PDF/SVG/EPS):
      pdf.fonttype = 42  → TrueType glyphs embedded, selectable in Illustrator/Inkscape
      ps.fonttype  = 42  → same for PostScript / EPS
      svg.fonttype = none → text elements, not paths
    """
    out_path = Path(out_path)
    out_dir = out_path.parent
    stem = out_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    editable_text = {
        "pdf.fonttype": 42,
        "ps.fonttype":  42,
        "svg.fonttype": "none",
    }
    with plt.rc_context(editable_text):
        for ext in formats:
            fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def hide_twinx_top_spine(ax2: plt.Axes) -> None:
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)
