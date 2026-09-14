import numpy as np
from data_generation_module import GraphNode, CycleNode, GraphEdge, UniversalGraphSystem, GeneSynthesizer


def test_generate_static_counts_shapes():
    nodes = [GraphNode('A', [0.0, 0.0], potential=-1.0), GraphNode('B', [2.0, 0.0], potential=-2.0)]
    edges = [GraphEdge(nodes[0], nodes[1])]
    system = UniversalGraphSystem(nodes, edges)
    gs = GeneSynthesizer(system, n_genes_total=10, seed=42)
    gs.add_trend_gene(edge_idx=0, n_genes=5, pattern='sigmoid')

    # generate on two points
    z = np.asarray([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    counts = gs.generate(z, simulate_counts=True)
    assert isinstance(counts, np.ndarray)
    assert counts.shape[0] == 2
    assert counts.shape[1] == 10


def test_generate_kinetics_shapes():
    nodes = [GraphNode('A', [0.0, 0.0], potential=-1.0), GraphNode('B', [2.0, 0.0], potential=-2.0)]
    edges = [GraphEdge(nodes[0], nodes[1])]
    system = UniversalGraphSystem(nodes, edges)
    gs = GeneSynthesizer(system, n_genes_total=10, seed=42)
    gs.add_trend_gene(edge_idx=0, n_genes=5, pattern='sigmoid')

    # small trajectory: 3 steps, 2 cells
    traj = np.stack([
        np.asarray([[0.0, 0.0], [0.2, 0.0]], dtype=np.float32),
        np.asarray([[0.5, 0.0], [1.0, 0.0]], dtype=np.float32),
        np.asarray([[1.5, 0.0], [2.0, 0.0]], dtype=np.float32)
    ])
    out = gs.generate_kinetics(traj, dt=0.1, simulate_counts=False)
    assert 'spliced' in out and 'unspliced' in out and 'alpha' in out and 'velocity' in out
    assert out['spliced'].shape[1] == 10
    assert out['spliced'].shape[0] == 3 * 2
