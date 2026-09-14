"""
graph_core.py - 通用图系统物理引擎

核心设计理念：万物皆节点 (Everything is a Node)

所有节点类型（GraphNode, CycleNode, SkeletonNode）都继承自 BaseNode，
通过统一的 get_potential_and_flow 接口向物理引擎提供：
1. 势能场 (Potential Field): 告诉粒子哪里的能量低
2. 流场 (Flow Field): 告诉粒子该往哪个方向流

这使得系统具有真正的多态性，用户可以轻松定义任意形状的节点。

使用示例：
    # 点吸引子节点
    stem = GraphNode("Stem", [-2.0, 0], potential=-2.0, sigma=0.3)

    # 环形节点
    prog = CycleNode("Prog", [0.0, 0], radius=1.0, omega=5.0)

    # 骨架节点（8字型）
    import numpy as np
    t = np.linspace(0, 2*np.pi, 100)
    x = np.sin(t)
    y = np.sin(2*t) / 2
    skeleton = np.stack([x, y], axis=1)
    fig8 = SkeletonNode("Figure8", skeleton, flow_strength=3.0)

    # 创建系统
    system = UniversalGraphSystem([stem, prog, fig8], edges=[...])
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Union, Optional
from abc import ABC, abstractmethod

# ==============================================================================
# 1. 统一节点接口 (Unified Node Interface)
# ==============================================================================

class BaseNode(ABC):
    """
    抽象基类：万物皆节点 (Everything is a Node)
    所有节点类型必须实现 get_potential_and_flow 方法
    提供统一的势能场和流场接口
    """
    def __init__(self, id: str, pos: List[float],
                 potential: float = -2.0, sigma: float = 0.3,
                 basin_strength: float = 0.0, basin_sigma: float = 0.3):
        self.id = id
        self.pos = pos
        self.potential = potential
        self.sigma = sigma
        self.basin_strength = basin_strength
        self.basin_sigma = basin_sigma

    @abstractmethod
    def get_potential_and_flow(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        核心接口：返回节点的势能场和流场

        Args:
            z: 粒子位置 [N, D]，N为粒子数，D为维度

        Returns:
            d2: 到节点骨架的平方距离 [N]
            u_center: 中心势能值 [N] 或标量
            force: 流场（非保守力） [N, D]
            sigma: 特征宽度 [N] 或标量
        """
        pass

    def get_projection_phase(self, z: np.ndarray) -> Optional[np.ndarray]:
        """
        投影相位接口：将高维坐标映射为标量相位

        Args:
            z: 粒子位置 [N, D]，N为粒子数，D为维度

        Returns:
            phase: 相位 [N]，如果节点不支持相位则返回 None
            - CycleNode: $\phi \in [0, 2\pi]$ (角度)
            - SkeletonNode: $\phi \in [0, 1]$ (归一化弧长) 或 $[0, 2\pi]$ (闭合回路)
            - GraphNode: None (点节点无相位概念)

        用途：用于基因表达的周期性模式生成
        """
        return None

@dataclass
class GraphNode(BaseNode):
    """点吸引子节点 (Point Attractor Node)"""
    id: str
    pos: List[float]
    potential: float = -2.0
    sigma: float = 0.3
    basin_strength: float = 0.0
    basin_sigma: float = 0.3

    def __post_init__(self):
        # 将dataclass转换为BaseNode的初始化方式
        BaseNode.__init__(self, self.id, self.pos, self.potential,
                         self.sigma, self.basin_strength, self.basin_sigma)

    def get_potential_and_flow(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        点吸引子的势能场和流场
        - 势能：到中心点的距离
        - 流场：无（返回零向量）
        """
        center = np.asarray(self.pos, dtype=z.dtype)
        diff = z - center
        d2 = (diff ** 2).sum(axis=1)  # [N]
        u_center = np.full_like(d2, self.potential)  # [N]
        force = np.zeros_like(z)  # [N, D]
        sigma = np.full_like(d2, self.sigma)  # [N]
        return d2, u_center, force, sigma

    def get_projection_phase(self, z: np.ndarray) -> Optional[np.ndarray]:
        """
        GraphNode 不支持相位（点节点无相位概念）
        """
        return None

@dataclass
class CycleNode(BaseNode):
    """环形节点 (支持指定高维旋转平面)"""
    id: str
    pos: List[float]
    potential: float = -2.0
    sigma: float = 0.3
    basin_strength: float = 0.0
    basin_sigma: float = 0.3
    radius: float = 1.0
    omega: float = 3.0
    clockwise: bool = False
    checkpoints: List[Tuple[float, float, float]] = field(default_factory=list)
    rotation_axes: Tuple[int, int] = (0, 1)

    def __post_init__(self):
        BaseNode.__init__(self, self.id, self.pos, self.potential,
                         self.sigma, self.basin_strength, self.basin_sigma)

    def get_potential_and_flow(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        环形吸引子的势能场和流场
        - 势能：到圆环的最近距离
        - 流场：沿切线方向的旋转力
        """
        center = np.asarray(self.pos, dtype=z.dtype)
        diff = z - center

        # 1. 分离变量：平面分量 vs 垂直分量
        ax1, ax2 = self.rotation_axes

        # 平面分量 (维持半径 R)
        diff_u = diff[:, ax1]
        diff_v = diff[:, ax2]
        dist_plane = np.sqrt(diff_u**2 + diff_v**2 + 1e-6)
        d_plane = dist_plane - self.radius

        # 垂直分量 (维持平面性，压回 0)
        all_axes = np.arange(z.shape[1])
        ortho_mask = (all_axes != ax1) & (all_axes != ax2)
        if ortho_mask.any():
            diff_ortho = diff[:, ortho_mask]
            d_ortho = np.linalg.norm(diff_ortho, axis=1)
        else:
            d_ortho = np.zeros_like(d_plane)

        # 总几何势能距离 (游泳圈形状)
        d2 = d_plane**2 + d_ortho**2

        # 2. 角度与检查点
        theta = np.arctan2(diff_v, diff_u)
        theta = np.where(theta < 0, theta + 2 * np.pi, theta)

        u_base = self.potential
        u_barrier = np.zeros_like(theta)

        if self.checkpoints:
            for phase, height, width in self.checkpoints:
                delta = np.abs(theta - phase)
                delta = np.minimum(delta, 2 * np.pi - delta)
                u_barrier += height * np.exp(-0.5 * (delta / width) ** 2)

        u_center = u_base + u_barrier

        # 3. 旋转力 (仅在平面内施加，沿切线方向)
        tangent_u = -diff_v
        tangent_v = diff_u
        norm = np.sqrt(tangent_u**2 + tangent_v**2 + 1e-6)
        tangent_u = tangent_u / norm
        tangent_v = tangent_v / norm

        if self.clockwise:
            tangent_u = -tangent_u
            tangent_v = -tangent_v

        force = np.zeros_like(z)
        force[:, ax1] = tangent_u * self.omega
        force[:, ax2] = tangent_v * self.omega

        sigma = np.full_like(d2, self.sigma)

        return d2, u_center, force, sigma

    def get_projection_phase(self, z: np.ndarray) -> np.ndarray:
        """
        环形节点的投影相位：极坐标角度

        返回相位 $\phi \in [0, 2\pi]$，基于旋转平面的极坐标角度

        Args:
            z: 粒子位置 [N, D]

        Returns:
            phase: 相位 [N]，范围 [0, 2\pi]
        """
        center = np.asarray(self.pos, dtype=z.dtype)
        diff = z - center

        # 提取旋转平面分量
        ax1, ax2 = self.rotation_axes
        diff_u = diff[:, ax1]
        diff_v = diff[:, ax2]

        # 计算角度：$\phi = \text{atan2}(z_v - c_v, z_u - c_u)$
        theta = np.arctan2(diff_v, diff_u)
        # 归一化到 [0, 2\pi]
        phase = np.where(theta < 0, theta + 2 * np.pi, theta)

        return phase

@dataclass
class SkeletonNode(BaseNode):
    """
    通用骨架节点 (General Skeleton Node)
    基于离散点集定义任意形状的骨架轨迹
    自动计算到骨架的最近距离（势能）和切线方向（流场）

    示例：
        # 创建8字形骨架
        t = np.linspace(0, 2*np.pi, 100)
        x = np.sin(t)
        y = np.sin(2*t) / 2
        skeleton = np.stack([x, y], axis=1)
        node = SkeletonNode("Figure8", skeleton)
    """
    id: str
    skeleton_points: Union[np.ndarray, List]  # [M, D] 骨架轨迹的离散点
    potential: float = -2.0
    sigma: float = 0.3
    basin_strength: float = 0.0
    basin_sigma: float = 0.3
    flow_strength: float = 3.0  # 流场强度
    pos: Optional[List[float]] = field(default=None)  # 自动计算为骨架中心
    is_closed: bool = False  # 是否为闭合回路（如 Figure-8）

    def __post_init__(self):
        # 确保skeleton_points是numpy数组格式
        self.skeleton_points = np.asarray(self.skeleton_points, dtype=np.float32)

        # 如果没有提供pos，使用骨架的中心点
        if self.pos is None:
            self.pos = self.skeleton_points.mean(axis=0).tolist()

        BaseNode.__init__(self, self.id, self.pos, self.potential,
                         self.sigma, self.basin_strength, self.basin_sigma)

        # 预先计算累积弧长（用于相位投影）
        # 计算每个骨架段之间的距离
        M = self.skeleton_points.shape[0]
        if M > 1:
            # 段长度：||p_i - p_{i-1}||
            segments = self.skeleton_points[1:] - self.skeleton_points[:-1]
            segment_lengths = np.linalg.norm(segments, axis=1)  # [M-1]

            # 累积距离：L_i = sum_{j=1}^i ||p_j - p_{j-1}||
            self.cumulative_lengths = np.zeros(M, dtype=np.float32)
            self.cumulative_lengths[1:] = np.cumsum(segment_lengths, axis=0)
            self.total_length = float(self.cumulative_lengths[-1])

            # 如果是闭合回路，添加最后一段到起点的距离
            if self.is_closed and M > 2:
                closing_segment = np.linalg.norm(self.skeleton_points[-1] - self.skeleton_points[0])
                self.total_length += float(closing_segment)
        else:
            # 单点骨架，无弧长概念
            self.cumulative_lengths = np.zeros(1, dtype=np.float32)
            self.total_length = 0.0

    def get_potential_and_flow(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        骨架节点的势能场和流场
        - 势能：到骨架轨迹的最近距离
        - 流场：沿骨架轨迹切线方向的流场

        使用向量化操作提高效率
        """
        skeleton = self.skeleton_points
        M, D = skeleton.shape
        N = z.shape[0]

        # 1. 计算到所有骨架点的距离，找到最近点和最近距离
        # z: [N, D], skeleton: [M, D]
        z_expanded = z[:, None, :]  # [N, 1, D]
        skeleton_expanded = skeleton[None, :, :]  # [1, M, D]

        # 计算所有点到所有骨架点的距离
        distances_sq = ((z_expanded - skeleton_expanded) ** 2).sum(axis=2)  # [N, M]

        # 找到最近点的索引
        nearest_idx = np.argmin(distances_sq, axis=1)  # [N]

        # 最近距离的平方
        d2 = distances_sq[np.arange(N), nearest_idx]  # [N]

        # 2. 计算势能
        u_center = np.full_like(d2, self.potential)  # [N]

        # 3. 计算流场：沿骨架轨迹的切线方向（向量化实现）
        # 计算所有骨架段的切线方向
        # 前向差分：skeleton[1:] - skeleton[:-1] -> [M-1, D]
        tangents_forward = skeleton[1:] - skeleton[:-1]  # [M-1, D]

        # 创建切线向量数组 [M, D]
        tangents = np.zeros((M, D), dtype=skeleton.dtype)
        tangents[0] = tangents_forward[0]  # 第一个点用第一段
        tangents[-1] = tangents_forward[-1]  # 最后一个点用最后一段
        tangents[1:-1] = (tangents_forward[:-1] + tangents_forward[1:]) / 2  # 中间点用前后段的平均

        # 归一化切线向量
        tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)  # [M, 1]
        tangent_norms = np.clip(tangent_norms, 1e-6, None)  # 避免除零
        tangents_unit = tangents / tangent_norms  # [M, D]

        # 使用索引选择每个粒子对应的切线方向
        force = tangents_unit[nearest_idx] * self.flow_strength  # [N, D]

        sigma = np.full_like(d2, self.sigma)  # [N]

        return d2, u_center, force, sigma

    def get_projection_phase(self, z: np.ndarray) -> np.ndarray:
        """
        骨架节点的投影相位：归一化弧长参数化

        原理：对于任意形状（如 8 字、螺旋），使用"弧长参数化" (Arc Length Parameterization)
        1. 找到每个粒子到骨架的最近点索引 k
        2. 使用该点的累积弧长 L_k
        3. 归一化：$\phi = L_k / L_{total}$ (范围 [0, 1])
        4. 如果 is_closed=True，映射到 $[0, 2\pi]$ (适用于 Figure-8 等闭合回路)

        Args:
            z: 粒子位置 [N, D]

        Returns:
            phase: 相位 [N]
            - 如果 is_closed=False: $\phi \in [0, 1]$ (归一化弧长)
            - 如果 is_closed=True: $\phi \in [0, 2\pi]$ (闭合回路)
        """
        skeleton = self.skeleton_points
        M, D = skeleton.shape
        N = z.shape[0]

        if self.total_length < 1e-6:
            # 单点骨架或无长度，返回零相位
            return np.zeros(N, dtype=z.dtype)

        # 计算每个粒子到所有骨架点的距离，找到最近点索引
        z_expanded = z[:, None, :]  # [N, 1, D]
        skeleton_expanded = skeleton[None, :, :]  # [1, M, D]
        distances_sq = ((z_expanded - skeleton_expanded) ** 2).sum(axis=2)  # [N, M]
        nearest_idx = np.argmin(distances_sq, axis=1)  # [N]

        # 获取每个粒子对应的累积弧长
        arc_lengths = self.cumulative_lengths[nearest_idx]  # [N]

        # 归一化：$\phi = L_k / L_{total}$ (范围 [0, 1])
        phase_normalized = arc_lengths / (self.total_length + 1e-8)

        # 如果是闭合回路，映射到 [0, 2\pi]
        if self.is_closed:
            phase = phase_normalized * 2 * np.pi
        else:
            phase = phase_normalized

        return phase

# ==============================================================================
# 2. 智能连接边 (Smart Edge)
# ==============================================================================

class GraphEdge:
    def __init__(self, node_start: BaseNode,
                 node_end: BaseNode,
                 barrier_height: float = 0.0):
        """
        智能连接边
        兼容所有继承自BaseNode的节点类型
        """
        self.node_start = node_start
        self.node_end = node_end

        # 1. 获取原始坐标
        p_start_raw = np.asarray(node_start.pos, dtype=np.float32)
        p_end_raw = np.asarray(node_end.pos, dtype=np.float32)

        if len(p_start_raw) != len(p_end_raw):
            raise ValueError(f"Dimension mismatch: {len(p_start_raw)} vs {len(p_end_raw)}")

        # ======================================================================
        # 核心修正：基于子空间的智能停靠 (Subspace-aware Docking)
        # ======================================================================

        # --- 修正起点 (Start Docking) ---
        if isinstance(node_start, CycleNode):
            # CycleNode: 基于旋转平面的切点停靠
            # 1. 计算全维度向量
            vec_full = p_end_raw - p_start_raw

            # 2. 提取旋转平面分量
            ax1, ax2 = node_start.rotation_axes
            vec_plane = np.zeros_like(vec_full)
            vec_plane[ax1] = vec_full[ax1]
            vec_plane[ax2] = vec_full[ax2]

            # 3. 检查平面内是否有分量
            norm_plane = np.linalg.norm(vec_plane)
            if norm_plane < 1e-6:
                # 极端情况：连接方向完全垂直于环平面
                offset = np.zeros_like(vec_full)
                offset[ax1] = node_start.radius
                self.start = p_start_raw + offset
                print(f"Warning: Edge orthogonal to Cycle {node_start.id}, using default entry.")
            else:
                # 4. 在旋转平面上计算单位向量
                unit_plane = vec_plane / norm_plane
                # 停靠点 = 圆心 + 半径 * 平面内的方向
                self.start = p_start_raw + unit_plane * node_start.radius
        elif isinstance(node_start, SkeletonNode):
            # SkeletonNode: 最近邻停靠 (Nearest Neighbor Docking)
            # 找到骨架轨迹上最接近目标点(p_end_raw)的点作为停靠点
            skeleton = node_start.skeleton_points
            # 计算目标点到所有骨架点的距离
            distances_sq = ((skeleton - p_end_raw[None, :]) ** 2).sum(axis=1)
            nearest_idx = np.argmin(distances_sq)
            self.start = skeleton[nearest_idx]
        else:
            # GraphNode等其他节点类型：直接使用节点位置
            self.start = p_start_raw

        # --- 修正终点 (End Docking) ---
        if isinstance(node_end, CycleNode):
            # CycleNode: 基于旋转平面的切点停靠
            # 1. 计算全维度向量 (Start -> End)
            vec_full = p_end_raw - self.start

            # 2. 提取旋转平面分量
            ax1, ax2 = node_end.rotation_axes
            vec_plane = np.zeros_like(vec_full)
            vec_plane[ax1] = vec_full[ax1]
            vec_plane[ax2] = vec_full[ax2]

            # 3. 检查平面分量
            norm_plane = np.linalg.norm(vec_plane)
            if norm_plane < 1e-6:
                offset = np.zeros_like(vec_full)
                offset[ax1] = node_end.radius
                self.end = p_end_raw - offset
                print(f"Warning: Edge orthogonal to Cycle {node_end.id}, using default entry.")
            else:
                # 4. 在平面上计算切入点
                unit_plane = vec_plane / norm_plane
                self.end = p_end_raw - unit_plane * node_end.radius
        elif isinstance(node_end, SkeletonNode):
            # SkeletonNode: 最近邻停靠 (Nearest Neighbor Docking)
            # 找到骨架轨迹上最接近起点(self.start)的点作为停靠点
            skeleton = node_end.skeleton_points
            # 计算起点到所有骨架点的距离
            distances_sq = ((skeleton - self.start[None, :]) ** 2).sum(axis=1)
            nearest_idx = np.argmin(distances_sq)
            self.end = skeleton[nearest_idx]
        else:
            # GraphNode等其他节点类型：直接使用节点位置
            self.end = p_end_raw

        # ======================================================================
        # 常规初始化
        # ======================================================================
        self.vec = self.end - self.start
        self.length_sq = (self.vec ** 2).sum()

        # 属性插值
        self.u_start = node_start.potential
        self.u_end = node_end.potential
        self.sigma_start = node_start.sigma
        self.sigma_end = node_end.sigma
        self.barrier_height = barrier_height

    def get_projection_t(self, z: np.ndarray) -> np.ndarray:
        p = z - self.start
        vec = self.vec
        t_raw = (p * vec).sum(axis=1) / (self.length_sq + 1e-8)
        return t_raw

    def distance_sq(self, z: np.ndarray) -> np.ndarray:
        t = np.clip(self.get_projection_t(z), 0.0, 1.0)
        closest = self.start + t[:, None] * self.vec
        return ((z - closest) ** 2).sum(axis=1)

    def get_component_data(self, z: np.ndarray):
        t_raw = self.get_projection_t(z)
        t = np.clip(t_raw, 0.0, 1.0)

        curr_sigma = (1 - t) * self.sigma_start + t * self.sigma_end
        base_u = (1 - t) * self.u_start + t * self.u_end

        barrier = self.barrier_height * np.exp(-((t - 0.5) ** 2) / (2 * 0.15 ** 2))
        u_center = base_u + barrier

        d2 = self.distance_sq(z)
        force = np.zeros_like(z)

        return d2, u_center, force, curr_sigma

# ==============================================================================
# 3. 通用系统管理器 (支持 N-维旋转与约束)
# ==============================================================================

class UniversalGraphSystem:
    def __init__(self, nodes, edges, fusion_temp=0.01, gradient_clip: float = 50.0):
        self.nodes = nodes
        self.edges = list(edges)
        self.T = fusion_temp
        self.gradient_clip = float(gradient_clip)

    def get_node_basins(self, z: np.ndarray):
        """计算节点的高斯深坑势能（basin）"""
        u_nodes = np.zeros(z.shape[0], dtype=z.dtype)
        for node in self.nodes:
            if isinstance(node, BaseNode) and node.basin_strength > 0:
                center = np.asarray(node.pos, dtype=z.dtype)
                d2 = ((z - center) ** 2).sum(axis=1)
                u_i = -node.basin_strength * np.exp(-d2 / (2 * node.basin_sigma ** 2))
                u_nodes += u_i
        return u_nodes

    def compute_fused_field(self, z: np.ndarray):
        """
        融合所有节点和边的势能场和流场
        使用统一的节点接口，支持多态调用
        """
        d2_list, u_center_list, force_list, sigma_list = [], [], [], []

        # 处理所有边
        for edge in self.edges:
            d2, u, f, sig = edge.get_component_data(z)
            d2_list.append(d2); u_center_list.append(u); force_list.append(f); sigma_list.append(sig)

        # 处理所有节点 - 使用统一接口
        for node in self.nodes:
            if isinstance(node, BaseNode):
                # 使用统一的多态接口
                d2, u, f, sig = node.get_potential_and_flow(z)
                # 确保u是数组格式
                if isinstance(u, (int, float)):
                    u = np.full_like(d2, u)
                elif u.ndim == 0:
                    u = np.broadcast_to(u, d2.shape)
                d2_list.append(d2); u_center_list.append(u); force_list.append(f); sigma_list.append(sig)

        if not d2_list:
            return np.zeros(z.shape[0], dtype=z.dtype), np.zeros_like(z)

        # 堆叠所有组件
        d2_stack = np.stack(d2_list, axis=1)
        u_center_stack = np.stack(u_center_list, axis=1)
        force_stack = np.stack(force_list, axis=1)
        sigma_stack = np.stack(sigma_list, axis=1)

        # SoftMin融合：基于局部权重和深度
        local_weight = np.exp(-d2_stack / (2 * sigma_stack ** 2))
        u_local_stack = u_center_stack * local_weight

        depth_stack = -u_local_stack
        weights = self._softmax(depth_stack / self.T, axis=1)

        # 融合势能和流场
        u_fused = (weights * u_local_stack).sum(axis=1)
        f_fused = (weights[:, :, None] * force_stack * local_weight[:, :, None]).sum(axis=1)

        return u_fused, f_fused

    @staticmethod
    def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
        """数值稳定的 softmax"""
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / e.sum(axis=axis, keepdims=True)

    def _potential_total(self, z: np.ndarray) -> np.ndarray:
        """总势能（fused + basins），用于漂移场的梯度计算"""
        u_fused, _ = self.compute_fused_field(z)
        u_basins = self.get_node_basins(z)
        return u_fused + u_basins

    def get_drift(self, z: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """Return the numerical latent drift b_num(z) used by simulations.

        漂移场 = -∇势能 + 旋转流，使用中心差分（数值梯度）替代自动微分：
            ∇u ≈ (u(z+ε·e_d) - u(z-ε·e_d)) / (2ε)

        Args:
            z: 粒子位置 [N, D]
            eps: 中心差分步长（默认 1e-4）
        """
        z = np.asarray(z, dtype=np.float32)
        _, f_rot = self.compute_fused_field(z)

        # 数值梯度（float64 计算以保证精度）
        z64 = z.astype(np.float64)
        N, D = z64.shape
        grad64 = np.zeros_like(z64)
        for d in range(D):
            z_plus = z64.copy()
            z_minus = z64.copy()
            z_plus[:, d] += eps
            z_minus[:, d] -= eps
            grad64[:, d] = (self._potential_total(z_plus) - self._potential_total(z_minus)) / (2 * eps)

        grad = grad64.astype(np.float32)
        grad = np.clip(grad, -self.gradient_clip, self.gradient_clip)  # 梯度截断，防抖动

        return -grad + f_rot
        return -grad + f_rot

    def potential_numpy(self, grid: np.ndarray) -> np.ndarray:
        u, _ = self.compute_fused_field(grid)
        u += self.get_node_basins(grid)
        return u

    def get_downstream_edges(self, edge_idx: int) -> List[int]:
        """
        查找指定边的所有下游边索引 (递归查找)
        用于实现 "Lineage Gene" 的持续表达
        """
        target_edge = self.edges[edge_idx]
        # 终点节点 ID
        end_node = target_edge.node_end

        downstream_indices = []

        # 简单的广度优先搜索 (BFS)
        # 找到所有以 end_node 为起点的边
        queue = [end_node]
        visited_nodes = {end_node.id}

        while queue:
            curr_node = queue.pop(0)

            # 遍历所有边，找以 curr_node 为起点的
            for i, e in enumerate(self.edges):
                # 只有当这条边不是我们起始的那条边，且起点匹配时
                if e.node_start.id == curr_node.id:
                    if i not in downstream_indices:
                        downstream_indices.append(i)
                        # 将这条边的终点加入队列继续搜
                        if e.node_end.id not in visited_nodes:
                            visited_nodes.add(e.node_end.id)
                            queue.append(e.node_end)
        return downstream_indices
