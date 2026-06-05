"""
测试 SkeletonNode 的最近邻停靠功能
验证 GraphEdge 能够正确找到骨架轨迹上的最近点
"""

import torch
import numpy as np
from data_generation_module import GraphNode, SkeletonNode, GraphEdge, UniversalGraphSystem


def test_skeleton_docking():
    """测试 SkeletonNode 的停靠逻辑"""
    print("=" * 60)
    print("测试 SkeletonNode 最近邻停靠功能")
    print("=" * 60)
    
    # 1. 创建一个简单的8字形状骨架
    t = np.linspace(0, 2 * np.pi, 100)
    x = 2.0 * np.sin(t)
    y = 2.0 * np.sin(2 * t) / 2
    skeleton = np.stack([x, y], axis=1)
    
    # 2. 创建节点
    start_node = GraphNode("Start", [-4.0, 0.0], potential=-1.0, sigma=0.3)
    fig8_node = SkeletonNode(
        id="Figure8",
        skeleton_points=skeleton,
        potential=-3.0,
        sigma=0.4,
        flow_strength=5.0
    )
    end_node = GraphNode("End", [4.0, 0.0], potential=-4.0, sigma=0.3)
    
    # 3. 创建边（测试停靠逻辑）
    edge1 = GraphEdge(start_node, fig8_node)  # Start -> Figure8
    edge2 = GraphEdge(fig8_node, end_node)    # Figure8 -> End
    
    print(f"\n边1 (Start -> Figure8):")
    print(f"  起点: {start_node.pos}")
    print(f"  修正后的起点: {edge1.start.tolist()}")
    print(f"  Figure8中心位置: {fig8_node.pos}")
    print(f"  修正后的终点（骨架最近点）: {edge1.end.tolist()}")
    
    # 验证终点确实在骨架轨迹上（或非常接近）
    skeleton_tensor = torch.tensor(skeleton, dtype=torch.float32)
    distances_to_skeleton = torch.norm(skeleton_tensor - edge1.end, dim=1)
    min_distance = distances_to_skeleton.min().item()
    print(f"  终点到骨架的最小距离: {min_distance:.6f} (应该接近0)")
    
    print(f"\n边2 (Figure8 -> End):")
    print(f"  修正后的起点（骨架最近点）: {edge2.start.tolist()}")
    print(f"  End位置: {end_node.pos}")
    print(f"  修正后的终点: {edge2.end.tolist()}")
    
    # 验证起点确实在骨架轨迹上
    distances_to_skeleton2 = torch.norm(skeleton_tensor - edge2.start, dim=1)
    min_distance2 = distances_to_skeleton2.min().item()
    print(f"  起点到骨架的最小距离: {min_distance2:.6f} (应该接近0)")
    
    # 4. 创建系统
    system = UniversalGraphSystem(
        nodes=[start_node, fig8_node, end_node],
        edges=[edge1, edge2]
    )
    
    print("\n✓ 停靠逻辑测试完成！")
    print("✓ 系统创建成功，可以用于轨迹模拟")
    
    return system


if __name__ == '__main__':
    test_skeleton_docking()
