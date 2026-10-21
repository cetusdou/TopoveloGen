import unittest

import numpy as np

from data_generation_module import GeneSynthesizer, build_topology
from data_generation_module.observation import draw_capture_factor, sample_counts


class ObservationTests(unittest.TestCase):
    def setUp(self):
        system, _ = build_topology("linear")
        self.synth = GeneSynthesizer(
            system, n_genes_total=2, lib_size_mean=100, lib_size_sd=0.2,
            dropout_prob=0.2,
        )
        self.abundance = np.array([[1.0, 2.0], [3.0, 4.0]])

    def test_capture_matches_existing_method(self):
        np.random.seed(7)
        expected = self.synth._draw_capture_factor(self.abundance)
        np.random.seed(7)
        actual = draw_capture_factor(self.abundance, 100, 0.2)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, (2, 1))

    def test_count_modes_match_existing_method(self):
        baseline = {
            "poisson": [[0.0, 4.0], [1.0, 5.0]],
            "nb": [[1.0, 1.0], [1.0, 4.0]],
            "unknown": [[0.0, 2.0], [3.0, 4.0]],
        }
        for mode in ("poisson", "nb", "unknown"):
            with self.subTest(mode=mode):
                self.synth.sampling = mode
                capture = np.ones((2, 1))
                np.random.seed(7)
                expected = self.synth._sample_counts(self.abundance, capture)
                np.random.seed(7)
                actual = sample_counts(
                    self.abundance, capture, 1.0, mode, self.synth.nb_dispersion,
                    self.synth.dropout_prob, self.synth.dropout_slope,
                )
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(actual, baseline[mode])

    def test_full_dropout(self):
        np.random.seed(7)
        counts = sample_counts(self.abundance, np.ones((2, 1)), 1.0,
                               "poisson", 1.0, 1.0, 0.0)
        np.testing.assert_array_equal(counts, np.zeros_like(counts))


if __name__ == "__main__":
    unittest.main()
