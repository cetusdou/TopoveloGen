import tempfile
import unittest
from pathlib import Path

from experiments.data_paths import pancreas_reference, repository_root, result_dir


class DataPathTests(unittest.TestCase):
    def test_default_input_is_inside_code_repository(self):
        expected = repository_root() / "data" / "Pancreas" / "endocrinogenesis_day15.h5ad"
        self.assertEqual(pancreas_reference(require=False), expected)

    def test_explicit_input_and_missing_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "reference.h5ad"
            existing.touch()
            self.assertEqual(pancreas_reference(existing), existing)
            missing = Path(tmp) / "missing.h5ad"
            with self.assertRaisesRegex(FileNotFoundError, "missing.h5ad"):
                pancreas_reference(missing)

    def test_result_path_does_not_create_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = result_dir("example", Path(tmp))
            self.assertEqual(target, Path(tmp) / "example")
            self.assertFalse(target.exists())

    def test_figure4_loader_does_not_substitute_another_dataset(self):
        from experiments.figure4_series_pipeline import load_pancreas_reference

        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.h5ad"
            with self.assertRaisesRegex(FileNotFoundError, "missing.h5ad"):
                load_pancreas_reference(missing)

    def test_figure4_config_preserves_missing_explicit_path(self):
        from experiments.figures.fig4series_pancreas_stats.lib import core

        previous = core.PANCREAS_H5AD
        try:
            with tempfile.TemporaryDirectory() as tmp:
                missing = Path(tmp) / "requested.h5ad"
                core.configure(pipeline={"stage1": {"pancreas_h5ad": str(missing)}})
                self.assertEqual(core.PANCREAS_H5AD, missing)
        finally:
            core.PANCREAS_H5AD = previous


if __name__ == "__main__":
    unittest.main()
