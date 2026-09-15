"""Generate the editable Figure3Series overview figure.

The large figure is composed during plotting in ``reproduce_figure3_series.py``.
This wrapper intentionally avoids loading rendered PNG panels, so the PDF/SVG
outputs remain editable vector graphics.
"""


import os
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    os.environ.setdefault("NUMBA_CACHE_DIR", str(REPO_ROOT / ".numba_cache"))
    runpy.run_path(str(REPO_ROOT / "experiments" / "reproduce_figure3_series.py"), run_name="__main__")


if __name__ == "__main__":
    main()
