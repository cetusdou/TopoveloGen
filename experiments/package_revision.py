"""Package retained supplementary outputs without rewriting run provenance."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "build" / "revision_archive"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    numeric = ROOT / "results/revision_numerical_kinetics"
    old_manifest = json.loads((numeric / "output_manifest.json").read_text())
    changes = {}
    for name, expected in old_manifest.items():
        actual = digest(numeric / name)
        if actual != expected:
            if name != "README.md":
                raise RuntimeError(f"Unexpected numerical output change: {name}")
            changes[name] = {"run_hash": expected, "packaged_hash": actual}
    files = set()
    for directory in ("data_generation_module", "experiments"):
        files.update((ROOT / directory).rglob("*.py"))
    files.update((ROOT / "experiments").glob("*REVISION*.md"))
    files.update((ROOT / "experiments").glob("revision*README.md"))
    for name in ("revision_coordinate_projection", "revision_numerical_kinetics"):
        files.update(p for p in (ROOT / "results" / name).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts)
    manifest = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(files)}
    metadata = {
        "scope": "Local supplementary bundle, not public release or historical-figure reconstruction",
        "documentation_only_changes_since_numerical_run": changes,
        "sha256": manifest,
    }
    encoded = json.dumps(metadata, indent=2).encode()
    (OUT / "manifest.json").write_bytes(encoded)
    readme = """# Supplementary revision bundle

Contains saved coordinate/projection and numerical/kinetic outputs, source code
and SHA-256 inventory. Not anonymous, not uploaded, no public identifier.
Use Python 3.10 and versions recorded in each run's provenance.json.
Run experiments/revision_coordinate_projection.py or
experiments/revision_numerical_kinetics.py from the extracted code directory.
Reruns overwrite results: retain an untouched archive copy.

For exact historical coordinate source provenance, in a separate extracted copy,
replace data_generation_module/synthesizer_gene.py with the coordinate run's
source_snapshot/synthesizer_gene.py; replace
experiments/figures/fig_coord_velocity_consistency/lib/core.py with its
source_snapshot/coordinate_core.py. The experiment itself used synchronized
updates and same-state velocities.

The numerical run used the corrected production generator. Its exact script is
in its source_snapshot directory; the current script has a later METHODS prose
correction. Original output_manifest.json is preserved: its README hash predates
a documentation correction. manifest.json records the changed documentation
hash and current inventory. Scientific output and run provenance are unchanged.

Original Fig.3 YAML and Fig.4 input checksum are unavailable. Old unsupported
benchmark panels were withdrawn, not retrospectively validated by these data.
"""
    archive = OUT / "revision_supplement.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            z.write(p, p.relative_to(ROOT).as_posix())
        z.writestr("manifest.json", encoded)
        z.writestr("README_REVISION.md", readme)
    with zipfile.ZipFile(archive) as z:
        for name, expected in manifest.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected, name
    print(json.dumps({"archive": str(archive), "files": len(files),
                      "bytes": archive.stat().st_size, "sha256": digest(archive),
                      "documentation_changes": changes}, indent=2))


if __name__ == "__main__":
    main()
