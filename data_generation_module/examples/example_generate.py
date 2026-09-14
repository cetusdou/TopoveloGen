"""可运行示例：生成二维景观、轨迹与基因表达。"""

import numpy as np

from data_generation_module import (
    CycleNode,
    GeneSynthesizer,
    GraphEdge,
    GraphNode,
    GraphSimulator,
    GraphVisualizer,
    UniversalGraphSystem,
)


def main():
    nodes = [
        GraphNode("Stem", [-3.0, 0.0], potential=-1.0, sigma=0.3),
        CycleNode("Cycle", [0.0, 0.0], radius=1.2, omega=4.0, potential=-2.5, sigma=0.3),
        GraphNode("Fate", [3.0, 0.0], potential=-4.5, sigma=0.15),
    ]
    edges = [GraphEdge(nodes[0], nodes[1]), GraphEdge(nodes[1], nodes[2])]
    system = UniversalGraphSystem(nodes, edges)

    print("开始模拟轨迹...")
    traj = GraphSimulator.run(system, "Stem", n_cells=20, steps=200)

    print("开始生成表达...")
    synthesizer = GeneSynthesizer(system, n_genes_total=20, seed=42)
    synthesizer.add_trend_gene(edge_idx=0, n_genes=8, pattern="sigmoid")
    synthesizer.add_cycle_gene(node_id="Cycle", n_genes=6)

    z = traj.reshape(-1, 2).astype(np.float32)
    counts = synthesizer.generate(z, simulate_counts=False)
    print("counts shape:", counts.shape)

    GraphVisualizer.plot_traj(system, traj, title="示例景观")


if __name__ == "__main__":
    main()
