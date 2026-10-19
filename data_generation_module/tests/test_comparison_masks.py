"""Comparison populations must be identical within each displayed contrast."""

import numpy as np
import unittest

from experiments import revision_projection_closure as projection
from experiments import revision_three_method_benchmark as benchmark


class ComparisonMaskTests(unittest.TestCase):
    def test_three_method_common_cells_exclude_any_method_zero_vector(self):
        reference = np.array([[1., 0.], [1., 0.], [1., 0.]])
        predictions = {
            "scvelo": np.array([[1., 0.], [1., 0.], [1., 0.]]),
            "unitvelo": np.array([[1., 0.], [0., 0.], [1., 0.]]),
            "deepvelo": np.array([[1., 0.], [1., 0.], [1., 0.]]),
        }

        common, method_masks = benchmark.common_valid_cells(predictions, reference)

        np.testing.assert_array_equal(common, [True, False, True])
        np.testing.assert_array_equal(method_masks["scvelo"], [True, True, True])
        np.testing.assert_array_equal(method_masks["unitvelo"], [True, False, True])
        np.testing.assert_array_equal(method_masks["deepvelo"], [True, True, True])


    def test_projection_common_cells_exclude_any_operator_invalid_score(self):
        scores = {
            "gene": np.array([0.1, 0.2, 0.3, 0.4]),
            "pca_linear": np.array([0.5, np.nan, 0.7, 0.8]),
            "scvelo_native_umap": np.array([0.4, 0.5, np.nan, 0.6]),
        }

        common = projection.common_score_cells(scores)

        np.testing.assert_array_equal(common, [True, False, False, True])
        row = projection.metric_row("cycle", 11, "all", "pca_linear",
                                    scores["pca_linear"], scores["gene"], common)
        self.assertEqual(row["n_valid"], 2)
        self.assertTrue(np.isclose(row["mean_delta_vs_gene"], 0.4))


if __name__ == "__main__":
    unittest.main()
