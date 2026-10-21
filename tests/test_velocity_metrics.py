import unittest

import numpy as np

from experiments.common.velocity_metrics import cosine, pushforward
from experiments.common.projection import positive_cosine_neighbor_projection


class VelocityMetricsTests(unittest.TestCase):
    def test_cosine_excludes_zero_vectors(self):
        a = np.array([[1., 0.], [0., 0.], [1., 1.]])
        b = np.array([[1., 0.], [1., 0.], [-1., -1.]])
        scores = cosine(a, b)
        np.testing.assert_allclose(scores[[0, 2]], [1., -1.])
        self.assertTrue(np.isnan(scores[1]))

    def test_pushforward_of_uniform_growth_is_zero_in_proportions(self):
        state = np.array([[2., 3.]])
        velocity = np.array([[4., 6.]])
        proportion, log_proportion = pushforward(state, velocity)
        np.testing.assert_allclose(proportion, 0., atol=1e-12)
        np.testing.assert_allclose(log_proportion, 0., atol=1e-12)

    def test_neighbor_projection_uses_positive_alignment(self):
        state = np.array([[0., 0.], [1., 0.], [0., 1.]])
        embedding = state.copy()
        velocity = np.array([[1., 0.], [0., 1.], [1., 0.]])
        neighbors = np.array([[1, 2], [0, 2], [0, 1]])
        projected = positive_cosine_neighbor_projection(
            velocity, state, embedding, neighbors,
        )
        np.testing.assert_allclose(projected[0], [1., 0.])


if __name__ == "__main__":
    unittest.main()
