import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Union, Tuple
# 引入你的基础类
from .graph_core import GraphNode, CycleNode
from collections import deque
from .graph_core import UniversalGraphSystem

class AutoLayout:
    def __init__(self, target_dim: int = None, variance_threshold: float = 0.95, bounds: tuple = (-4.0, 4.0)):
        """
        Args:
            target_dim: 强制指定的物理维度。
            variance_threshold: 自动维度保留的方差比例。
            bounds: 物理坐标的缩放范围 (min, max)。
        """
        self.target_dim = target_dim
        self.variance_threshold = variance_threshold
        self.bounds = bounds
        
        # 核心模型
        self.pca = None
        self.scaler = None
        
    def fit_transform(self, nodes: List[GraphNode], expression_dict: Dict[str, np.ndarray]) -> int:
        """
        阶段 1: 仅使用"锚点"(理想稳态表达)来确定物理空间的骨架 (Topology Skeleton)
        """
        # 1. 构建表达量矩阵
        valid_nodes = []
        expr_matrix = []
        for node in nodes:
            if node.id in expression_dict:
                valid_nodes.append(node)
                expr_matrix.append(expression_dict[node.id])
            else:
                print(f"Warning: Node {node.id} has no anchor expression. Position unchanged.")
        
        if len(valid_nodes) < 2:
            raise ValueError("Need at least 2 nodes to determine layout.")

        X = np.array(expr_matrix)
        
        # 2. 决定维度
        max_possible_dim = min(len(valid_nodes), X.shape[1])
        if self.target_dim is not None:
            n_components = min(self.target_dim, max_possible_dim)
        else:
            pca_temp = PCA(n_components=max_possible_dim)
            pca_temp.fit(X)
            cum_var = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cum_var >= self.variance_threshold) + 1
            print(f"Auto-selected dimension: {n_components} (Variance covered: {cum_var[n_components-1]:.2%})")

        # 3. 执行 PCA
        self.pca = PCA(n_components=n_components)
        coords = self.pca.fit_transform(X)
        
        # 4. 归一化到物理范围 (Bounds)
        self.scaler = MinMaxScaler(feature_range=self.bounds)
        coords_scaled = self.scaler.fit_transform(coords)
        
        # 5. 更新节点坐标
        for i, node in enumerate(valid_nodes):
            node.pos = coords_scaled[i].tolist()
            # print(f"Updated {node.id} pos: {np.round(node.pos, 2)}")
            
        return n_components

    def transform(self, X_sc: np.ndarray) -> np.ndarray:
        """
        功能: 将新的单细胞数据投影到 fit_transform 确定的物理空间中。
        Args:
            X_sc: [N_cells, N_genes] 的单细胞表达矩阵
        Returns:
            X_latent: [N_cells, N_dim] 物理空间坐标
        """
        if self.pca is None or self.scaler is None:
            raise RuntimeError("AutoLayout must be fitted (fit_transform) before calling transform.")
        
        # 1. PCA 投影
        coords = self.pca.transform(X_sc)
        # 2. 空间缩放 (使用之前确定好的骨架范围)
        coords_scaled = self.scaler.transform(coords)
        
        return coords_scaled

    def fit_physics_from_graph(self, 
                               system: UniversalGraphSystem, 
                               X_sc: np.ndarray = None, 
                               labels: np.ndarray = None,
                               level_drop: float = 5.0,     
                               density_bonus: float = 0.5,
                               sigma_scale: float = 1.0,
                               manual_overrides: Dict[str, Dict] = None):
        """
        全自动物理参数推断 + 自动同步 Edge
        
        Args:
            manual_overrides: 字典，用于强制指定某些节点的参数。
                              例如: {'Prog': {'potential': -5.0, 'sigma': 0.5}}
        """

        
        # 1. 构建图结构 & BFS 计算层级
        adj = {node.id: [] for node in system.nodes}
        in_degree = {node.id: 0 for node in system.nodes}
        for edge in system.edges:
            u, v = edge.node_start.id, edge.node_end.id
            adj[u].append(v)
            in_degree[v] += 1
            
        queue = deque([node_id for node_id, deg in in_degree.items() if deg == 0])
        levels = {node.id: 0 for node in system.nodes}
        if not queue and system.nodes: queue.append("Stem") # Fallback
        
        while queue:
            curr_id = queue.popleft()
            curr_level = levels[curr_id]
            for neighbor_id in adj[curr_id]:
                levels[neighbor_id] = max(levels[neighbor_id], curr_level + 1)
                queue.append(neighbor_id)
        
        # 2. 结合数据计算物理参数
        has_data = (X_sc is not None) and (labels is not None)
        X_latent = None
        max_count = 1
        if has_data:
            X_latent = self.transform(X_sc)
            unique, counts = np.unique(labels, return_counts=True)
            max_count = counts.max()
            
        for node in system.nodes:
            # A. 基础势能 (Global Hierarchy)
            lvl = levels.get(node.id, 0)
            u_global = -1.0 - (lvl * level_drop)
            
            # B. 局部密度修正 (Data Driven)
            u_local = 0.0
            sigma_val = 0.5 
            
            if has_data and node.id in labels:
                mask = (labels == node.id)
                if np.any(mask):
                    cluster_data = X_latent[mask]
                    rel_density = len(cluster_data) / max_count
                    u_local = -density_bonus * rel_density # 密度微调
                    
                    center = np.array(node.pos)
                    rmse = np.sqrt(np.mean(np.sum((cluster_data - center)**2, axis=1)))
                    sigma_val = max(rmse * sigma_scale, 0.2)
            
            node.potential = u_global + u_local
            node.sigma = sigma_val
            
            # C. 应用手动修正 (Manual Overrides)
            # 这允许用户强制覆盖掉自动计算的不合理值 (如 CycleNode 的深坑)
            if manual_overrides and node.id in manual_overrides:
                overrides = manual_overrides[node.id]
                for key, val in overrides.items():
                    if hasattr(node, key):
                        setattr(node, key, val)
                        print(f"  [Override] Node '{node.id}' {key} set to {val}")

            # print(f"  Node '{node.id}': Level={lvl} => U={node.potential:.2f}, σ={node.sigma:.2f}")

        # ======================================================================
        # 3. [新增功能] 自动同步 Edge 属性
        # ======================================================================
        print("  -> Syncing Edges with updated Node physics...")
        for edge in system.edges:
            # 强制刷新 Edge 的势能和宽度，使其与 Node 保持一致
            edge.u_start = edge.node_start.potential
            edge.u_end = edge.node_end.potential
            edge.sigma_start = edge.node_start.sigma
            edge.sigma_end = edge.node_end.sigma
            
            # 如果是 CycleNode 连接出的边，可能需要特殊处理 (可选)
            # 但通常上面的同步就足够解决"看不见的墙"的问题了
            
        # print("--- Physics Fitting Complete ---\n")