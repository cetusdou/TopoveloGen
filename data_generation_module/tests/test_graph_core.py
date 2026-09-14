import numpy as np
from data_generation_module import GraphNode, CycleNode, GraphEdge, UniversalGraphSystem


def test_graph_edge_docking():
    # Simple case: 2D nodes
    n1 = GraphNode('A', [0.0, 0.0], potential=-1.0, sigma=0.2)
    n2 = GraphNode('B', [2.0, 0.0], potential=-2.0, sigma=0.2)
    e = GraphEdge(n1, n2)
    # start and end should be equal to node coords for non-cycle nodes
    assert np.allclose(e.start, np.asarray([0.0, 0.0], dtype=np.float32))
    assert np.allclose(e.end, np.asarray([2.0, 0.0], dtype=np.float32))


def test_cycle_docking_and_projection():
    center = [0.0, 0.0]
    c = CycleNode('C', pos=center, radius=1.0, omega=1.0, rotation_axes=(0, 1), potential=-2.0)
    n = GraphNode('N', [3.0, 0.0], potential=-1.0)
    edge = GraphEdge(c, n)
    # Start of edge should land approximately on the cycle rim (distance ~ radius)
    dist = np.linalg.norm(edge.start - np.asarray(center, dtype=np.float32))
    assert np.isclose(float(dist), 1.0, rtol=1e-3)

    # projection of a point on the line should produce t in [0, 1]
    z = np.asarray([[edge.start[0], edge.start[1]]], dtype=np.float32)
    t = edge.get_projection_t(z)
    assert 0.0 <= float(t[0]) <= 1.0

