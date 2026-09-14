from typing import Tuple

import numpy as np

from .graph_core import CycleNode, GraphEdge, GraphNode, UniversalGraphSystem
from .synthesizer_gene import GeneSynthesizer


def build_topology(trajectory_type: str = "cycle_bifurcation") -> Tuple[UniversalGraphSystem, str]:
    """
    Build a small preset topology and return (system, start_node_id).
    """
    if trajectory_type == "linear":
        nodes = [
            GraphNode("Stem", [-2.0, 0.0], potential=-1.0, sigma=0.8),
            GraphNode("Prog", [0.0, 0.0], potential=-2.0, sigma=0.6),
            GraphNode("Fate", [2.0, 0.0], potential=-4.0, sigma=0.5),
        ]
        edges = [GraphEdge(nodes[0], nodes[1]), GraphEdge(nodes[1], nodes[2])]
        return UniversalGraphSystem(nodes, edges, fusion_temp=0.05), "Stem"

    if trajectory_type == "bifurcation":
        nodes = [
            GraphNode("Stem", [-2.0, 0.0], potential=-1.0, sigma=0.8),
            GraphNode("Prog", [0.0, 0.0], potential=-2.0, sigma=0.5),
            GraphNode("FateA", [2.0, 1.5], potential=-5.0, sigma=0.5, basin_sigma=0.5, basin_strength=0.6),
            GraphNode("FateB", [2.0, -1.5], potential=-5.0, sigma=0.5, basin_sigma=0.5, basin_strength=0.6),
        ]
        edges = [
            GraphEdge(nodes[0], nodes[1]),
            GraphEdge(nodes[1], nodes[2]),
            GraphEdge(nodes[1], nodes[3]),
        ]
        return UniversalGraphSystem(nodes, edges, fusion_temp=0.05), "Stem"

    if trajectory_type == "cycle":
        nodes = [
            GraphNode("Stem", [-2.0, 0.0], potential=-1.0, sigma=0.8),
            CycleNode("Cycle", [0.0, 0.0], potential=-3.0, sigma=0.5, radius=1.0, omega=5.0, clockwise=False),
            GraphNode("Fate", [2.0, 0.0], potential=-4.0, sigma=0.5, basin_sigma=0.6, basin_strength=0.55),
        ]
        edges = [GraphEdge(nodes[0], nodes[1]), GraphEdge(nodes[1], nodes[2])]
        return UniversalGraphSystem(nodes, edges, fusion_temp=0.12), "Stem"

    # default: cycle_bifurcation
    nodes = [
        GraphNode("Stem", [-2.0, 0.0], potential=0.0, sigma=0.8, basin_sigma=0.5, basin_strength=0.2),
        CycleNode("Prog", [0.0, 0.0], potential=-1.5, sigma=0.5, radius=1.0, omega=5.0, clockwise=False),
        GraphNode("FateA", [2.0, 1.5], potential=-4.0, sigma=0.5, basin_sigma=0.6, basin_strength=0.5),
        GraphNode("FateB", [2.0, -1.5], potential=-4.0, sigma=0.5, basin_sigma=0.6, basin_strength=0.5),
    ]
    edges = [
        GraphEdge(nodes[0], nodes[1]),
        GraphEdge(nodes[1], nodes[2]),
        GraphEdge(nodes[1], nodes[3]),
    ]
    return UniversalGraphSystem(nodes, edges, fusion_temp=0.12), "Stem"


def attach_default_gene_modules(
    synthesizer: GeneSynthesizer,
    trajectory_type: str,
    n_genes: int,
) -> None:
    """
    Register a default set of modules for the chosen topology.
    """
    if trajectory_type in {"linear", "cycle"}:
        fracs = np.array([0.35, 0.35, 0.30])
    else:
        fracs = np.array([0.30, 0.20, 0.20, 0.20, 0.10])

    counts = (fracs * n_genes).astype(int)
    counts[0] += n_genes - int(counts.sum())

    if trajectory_type == "linear":
        synthesizer.add_trend_gene(edge_idx=0, n_genes=int(counts[0]), pattern="sigmoid", include_cycle_rim=False)
        synthesizer.add_trend_gene(edge_idx=1, n_genes=int(counts[1]), pattern="sigmoid", include_cycle_rim=False)
        synthesizer.add_trend_gene(edge_idx=0, n_genes=int(counts[2]), pattern="impulse", reverse=True, include_cycle_rim=False)
        return

    if trajectory_type == "bifurcation":
        synthesizer.add_trend_gene(edge_idx=0, n_genes=int(counts[0]), pattern="sigmoid", include_cycle_rim=False)
        synthesizer.add_trend_gene(edge_idx=1, n_genes=int(counts[2]), pattern="sigmoid", include_cycle_rim=False)
        synthesizer.add_trend_gene(edge_idx=2, n_genes=int(counts[3]), pattern="sigmoid", include_cycle_rim=False)
        synthesizer.add_trend_gene(edge_idx=0, n_genes=int(counts[4]), pattern="impulse", reverse=True, include_cycle_rim=False)
        return

    if trajectory_type == "cycle":
        synthesizer.add_trend_gene(edge_idx=0, n_genes=int(counts[0]), pattern="sigmoid")
        synthesizer.add_cycle_gene(node_id="Cycle", n_genes=int(counts[1]))
        synthesizer.add_trend_gene(edge_idx=1, n_genes=int(counts[2]), pattern="sigmoid")
        return

    # default: cycle_bifurcation
    synthesizer.add_trend_gene(edge_idx=0, n_genes=int(counts[0]), pattern="sigmoid")
    synthesizer.add_cycle_gene(node_id="Prog", n_genes=int(counts[1]))
    synthesizer.add_trend_gene(edge_idx=1, n_genes=int(counts[2]), pattern="sigmoid")
    synthesizer.add_trend_gene(edge_idx=2, n_genes=int(counts[3]), pattern="sigmoid")
    synthesizer.add_trend_gene(edge_idx=0, n_genes=int(counts[4]), pattern="impulse", reverse=True)
