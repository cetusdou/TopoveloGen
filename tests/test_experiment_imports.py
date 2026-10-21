import unittest

from experiments.common.coordinate_simulation import simulate, infer_and_smooth
from experiments import revision_coordinate_projection as coordinate


class ExperimentImportTests(unittest.TestCase):
    def test_coordinate_runner_reexports_shared_operations(self):
        self.assertIs(coordinate.simulate, simulate)
        self.assertIs(coordinate.infer_and_smooth, infer_and_smooth)

    def test_small_coordinate_simulation(self):
        config = dict(dt=.01, steps=6, burnin_steps=0, diffusion=.006,
                      trajectories=8, genes=80, observations=10,
                      save_every=2, neighbors=3)
        data = simulate(config, "cycle", 11)
        self.assertEqual(data["z"].shape, (10, 2))
        self.assertEqual(data["v"].shape, (10, 80))
        self.assertEqual(data["sc"].shape, (10, 80))


if __name__ == "__main__":
    unittest.main()
