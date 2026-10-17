"""Guard revision experiment caches against mixed configurations."""
from __future__ import annotations

import json
from pathlib import Path


def prepare_cache(output: Path, config: dict) -> None:
    """Create an output archive or verify its existing provenance exactly."""
    normalized = json.loads(json.dumps(config))
    config_path = output / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != normalized:
            raise RuntimeError(
                f"Cached results in {output} use different settings or source files. "
                "Choose a fresh --out directory; existing results were not changed."
            )
        return
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"Output directory {output} is not empty but has no config.json. "
            "Choose a fresh --out directory; existing files were not changed."
        )
    output.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
