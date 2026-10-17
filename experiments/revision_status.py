"""Read-only audit of archived revision source hashes.

This reports whether the present checkout is byte-identical to the source
files recorded with each archive. A mismatch does not invalidate archived
results; it means a rerun under this checkout is a different run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAMES = ("config.json", "provenance.json", "input_config.json", "results.json")


def recorded_sources(manifest: Path, record: dict) -> dict[str, str]:
    sources = {}
    records = [(record, f"experiments/{manifest.parent.name}.py")]
    archived = record.get("archive_configuration")
    if isinstance(archived, dict):
        records.append((archived, "experiments/revision_coordinate_projection.py"))
    for item, script in records:
        for field in ("source_sha256", "source_hashes", "script_sha256"):
            value = item.get(field)
            if isinstance(value, dict):
                sources.update(value)
            elif field in ("source_sha256", "script_sha256") and isinstance(value, str):
                sources[script] = value
    simulator_hash = record.get("simulator_source_sha256")
    if isinstance(simulator_hash, str):
        sources["experiments/figures/fig_coord_velocity_consistency/lib/core.py"] = simulator_hash
    return sources


def inspect(manifest: Path) -> tuple[str, int, list[str]]:
    record = json.loads(manifest.read_text(encoding="utf-8"))
    sources = recorded_sources(manifest, record)
    if not sources:
        return "NO_SOURCE_HASH", 0, []
    changed = []
    for name, expected in sources.items():
        source = ROOT / Path(name.replace("\\", "/"))
        if not source.is_file():
            changed.append(f"missing {name}")
        elif hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            changed.append(f"changed {name}")
    return ("MATCH" if not changed else "DIFFERENT"), len(sources), changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        help="Exit nonzero if an archived source differs from this checkout")
    args = parser.parse_args()
    manifests = [path for folder in sorted((ROOT / "results").glob("revision_*"))
                 for name in MANIFEST_NAMES if (path := folder / name).is_file()]
    if not manifests:
        print("No revision manifests found.")
        return 1 if args.strict else 0
    archive_statuses: dict[Path, list[str]] = {}
    for manifest in manifests:
        status, count, changed = inspect(manifest)
        archive_statuses.setdefault(manifest.parent, []).append(status)
        print(f"{status:14} {manifest.relative_to(ROOT)} ({count} recorded sources)")
        for item in changed:
            print(f"  {item}")
    incomplete = any("DIFFERENT" in statuses or all(status == "NO_SOURCE_HASH" for status in statuses)
                     for statuses in archive_statuses.values())
    return 1 if args.strict and incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
