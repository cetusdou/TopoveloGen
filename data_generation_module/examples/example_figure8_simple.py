"""
8 字形节点快速示例。
"""

import numpy as np

from data_generation_module import GraphVisualizer, SkeletonNode, UniversalGraphSystem

t = np.linspace(0, 2 * np.pi, 200)
x = 2.0 * np.sin(t)
y = 2.0 * np.sin(2 * t) / 2
skeleton = np.stack([x, y], axis=1)

fig8_node = SkeletonNode(
    id="Figure8",
    skeleton_points=skeleton,
    potential=-3.0,
    sigma=0.4,
    flow_strength=5.0,
)

system = UniversalGraphSystem(nodes=[fig8_node], edges=[])
GraphVisualizer.plot_landscape(system, title="8字形状节点", bounds=(-3, 3), show_skeleton=True, show=True)

print("✓ 8字形状节点创建完成")
print(f"节点位置: {fig8_node.pos}")
print(f"骨架点数: {len(skeleton)}")
