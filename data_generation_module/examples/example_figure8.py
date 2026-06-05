"""
8 字形 SkeletonNode 示例。
展示如何创建任意骨架形状并进行轨迹模拟与可视化。
"""

import numpy as np
import torch

from data_generation_module import (
    GraphEdge,
    GraphNode,
    GraphSimulator,
    GraphVisualizer,
    SkeletonNode,
    UniversalGraphSystem,
)


def create_figure8_skeleton(n_points=200, scale=2.0):
    t = np.linspace(0, 2 * np.pi, n_points)
    x = scale * np.sin(t)
    y = scale * np.sin(2 * t) / 2
    skeleton = np.stack([x, y], axis=1)
    return torch.tensor(skeleton, dtype=torch.float32)


def example_simple_figure8():
    skeleton = create_figure8_skeleton(n_points=200, scale=2.0)
    fig8_node = SkeletonNode(
        id="Figure8",
        skeleton_points=skeleton,
        potential=-3.0,
        sigma=0.4,
        flow_strength=5.0,
    )
    system = UniversalGraphSystem(nodes=[fig8_node], edges=[])
    GraphVisualizer.plot_landscape(system, title="8字形势能场", bounds=(-3, 3), show_skeleton=True, show=True)
    return system, fig8_node


def example_figure8_with_trajectories():
    skeleton = create_figure8_skeleton(n_points=300, scale=2.5)
    fig8_node = SkeletonNode(
        id="Figure8",
        skeleton_points=skeleton,
        potential=-3.5,
        sigma=0.35,
        flow_strength=6.0,
    )
    system = UniversalGraphSystem(nodes=[fig8_node], edges=[])
    traj = GraphSimulator.run(system, start_node_id="Figure8", n_cells=30, steps=500, dt=0.02)
    GraphVisualizer.plot_traj(system, traj=traj, title="8字节点轨迹", bounds=(-4, 4), show_skeleton=True, show=True)
    return system, traj


def example_figure8_in_graph():
    skeleton = create_figure8_skeleton(n_points=250, scale=2.0)
    stem = GraphNode("Stem", [-4.0, 0.0], potential=-1.5, sigma=0.3)
    fig8 = SkeletonNode(
        id="Figure8",
        skeleton_points=skeleton,
        potential=-3.0,
        sigma=0.4,
        flow_strength=4.0,
    )
    fate_a = GraphNode("FateA", [3.0, 2.0], potential=-4.5, sigma=0.2)
    fate_b = GraphNode("FateB", [3.0, -2.0], potential=-4.5, sigma=0.2)
    edges = [GraphEdge(stem, fig8), GraphEdge(fig8, fate_a), GraphEdge(fig8, fate_b)]
    system = UniversalGraphSystem(nodes=[stem, fig8, fate_a, fate_b], edges=edges)
    traj = GraphSimulator.run(system, start_node_id="Stem", n_cells=50, steps=400)
    GraphVisualizer.plot_traj(system, traj=traj, title="Stem->Figure8->Fate", bounds=(-5, 5), show_skeleton=True, show=True)
    return system, traj


def main():
    example_simple_figure8()
    example_figure8_with_trajectories()
    example_figure8_in_graph()


if __name__ == "__main__":
    main()
