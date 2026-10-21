import unittest
import numpy as np
from data_generation_module import build_topology

from experiments.common.coordinate_simulation import simulate, infer_and_smooth
from experiments.common.native_projection import region_labels, fixed_distances, native_project
from experiments.common.numerical_controls import CONFIG, run_coupled, scaffold_distance
from experiments import revision_coordinate_projection as coordinate
from experiments import revision_projection_closure as projection
from experiments import revision_numerical_kinetics as numerical


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

    def test_projection_and_numerical_runners_reexport_shared_operations(self):
        self.assertIs(projection.region_labels, region_labels)
        self.assertIs(projection.fixed_distances, fixed_distances)
        self.assertIs(projection.native_project, native_project)
        self.assertIs(numerical.CONFIG, CONFIG)
        self.assertIs(numerical.run_coupled, run_coupled)
        self.assertIs(numerical.scaffold_distance, scaffold_distance)

    def test_cycle_region_labels(self):
        positions = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
        np.testing.assert_array_equal(
            region_labels("cycle", positions),
            ["quadrant_1", "quadrant_2", "quadrant_3", "quadrant_4"],
        )

    def test_fixed_neighbor_distances(self):
        data = dict(neighbor_indices=np.array([[1, 2], [0, 2], [0, 1]]),
                    sc=np.array([[1., 2.], [3., 4.], [5., 6.]]),
                    capture=np.ones((3, 1)))
        graph = fixed_distances(data)
        self.assertEqual(graph.shape, (3, 3))
        self.assertEqual(graph.nnz, 6)

    def test_coupled_path_and_scaffold_distance_shapes(self):
        system, start = build_topology("linear")
        initial = np.zeros((2, 2))
        increments = np.zeros((4, 2, 2))
        trajectory, discrepancy = run_coupled(system, start, initial, increments, .005)
        self.assertEqual(trajectory.shape, (5, 2, 2))
        self.assertEqual(scaffold_distance(system, trajectory).shape, (5, 2))
        self.assertEqual(discrepancy, 0.)


if __name__ == "__main__":
    unittest.main()
