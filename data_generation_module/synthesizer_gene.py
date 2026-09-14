import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any
import yaml

from .graph_core import UniversalGraphSystem, GraphEdge, CycleNode, GraphNode


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid（等价于 torch.sigmoid）"""
    return 1.0 / (1.0 + np.exp(-x))


class GeneSynthesizer:
    """
    基于拓扑投影与动力学的基因表达生成器

    Update:
    - 支持动力学模式 (Kinetic Mode): 生成 Spliced/Unspliced 矩阵用于 RNA Velocity。
    - 逻辑层级: 几何位置 -> 转录速率 (alpha) -> ODE 积分 -> 表达量 (u, s)
    """
    def __init__(
        self,
        system: UniversalGraphSystem,
        n_genes_total: int = 1000,
        edge_strength: float = 5.0,
        cycle_strength: float = 3.0,
        background_noise: float = 0.01,
        lib_size_mean: float = 10000.0,
        lib_size_sd: float = 0.2,
        sampling: str = 'poisson',  # 'poisson' or 'nb'
        nb_dispersion: float = 1.0,
        dropout_prob: float = 0.0,  # Dropout probability (0.0 = no dropout, typical: 0.1-0.3)
        dropout_slope: float = 0.1,  # Slope for dropout probability vs expression (higher = more dropout for low expression)
        unspliced_capture_ratio: float = 0.2,
        seed: Optional[int] = None,
        # --- 动力学参数 (新) ---
        splicing_rate: float = 1.0,   # beta
        degradation_rate: float = 0.5, # gamma
        grn_strength: float = 2.0,  # 调控信号的强度权重
        # --- 随机参数范围 ---
        sigmoid_center_range: tuple = (0.2, 0.8),
        sigmoid_slope_range: tuple = (5.0, 15.0),
        impulse_center_range: tuple = (0.1, 0.9),
        impulse_width_range: tuple = (0.05, 0.15),
        # --- 全局动力学衰减参数 (Global Kinetic Decay) ---
        enable_global_decay: bool = False,  # 是否启用全局衰减（模拟 Neuroblast -> Granule）
        global_decay_start_rate: float = 1.3,  # Neuroblast 期转录速率倍数（t=0）
        global_decay_end_rate: float = 0.6,    # Granule 期转录速率倍数（t=1）
    ):
        self.system = system
        self.n_genes = n_genes_total
        self.modules = []
        self.current_gene_idx = 0

        # 存储基因名称列表 [gene_0, gene_1, ...]
        self.gene_names = [None] * n_genes_total

        # 存储 Ground Truth (无噪声)
        self.expression_clean = None
        self.velocity_clean = None # 新增：真实的 RNA Velocity
        self.grn_strength = float(grn_strength)
        self.adj_matrix = None  # shape will be [n_genes, n_genes]
        # Hyper-parameters
        self.edge_strength = float(edge_strength)
        self.cycle_strength = float(cycle_strength)
        self.background_noise = float(background_noise)
        self.lib_size_mean = float(lib_size_mean)
        self.lib_size_sd = float(lib_size_sd)
        self.sampling = sampling
        self.nb_dispersion = float(nb_dispersion)
        self.dropout_prob = float(dropout_prob)
        self.dropout_slope = float(dropout_slope)
        self.unspliced_capture_ratio = float(unspliced_capture_ratio)
        self.seed = seed

        # Kinetics - 支持每个基因独立的 beta 和 gamma
        # 初始化为全局默认值，后续可以通过配置或方法设置每个基因的值
        self.beta = np.full((n_genes_total,), float(splicing_rate), dtype=np.float32)
        self.gamma = np.full((n_genes_total,), float(degradation_rate), dtype=np.float32)
        self.default_beta = float(splicing_rate)
        self.default_gamma = float(degradation_rate)

        # Random parameter ranges
        self.sigmoid_center_range = sigmoid_center_range
        self.sigmoid_slope_range = sigmoid_slope_range
        self.impulse_center_range = impulse_center_range
        self.impulse_width_range = impulse_width_range

        # Global kinetic decay parameters
        self.enable_global_decay = enable_global_decay
        self.global_decay_start_rate = float(global_decay_start_rate)
        self.global_decay_end_rate = float(global_decay_end_rate)

    def _allocate_indices(self, count):
        """分配基因索引"""
        if self.current_gene_idx + count > self.n_genes:
            raise ValueError(f"Exceeded total gene capacity ({self.n_genes})")
        indices = np.arange(self.current_gene_idx, self.current_gene_idx + count)
        self.current_gene_idx += count
        return indices

    # =====================================================
    # 1. 定义基因行为模块 (保持不变)
    # =====================================================
    def set_grn(self, adjacency_matrix: Union[np.ndarray, List]):
        """
        设置基因调控网络 (GRN)
        adj[j, i] > 0 表示基因 j 激活 基因 i
        adj[j, i] < 0 表示基因 j 抑制 基因 i
        """
        matrix = np.asarray(adjacency_matrix, dtype=np.float32)

        if matrix.shape != (self.n_genes, self.n_genes):
            raise ValueError(f"Adjacency matrix shape must be ({self.n_genes}, {self.n_genes})")

        # 归一化矩阵，防止数值爆炸 (可选但推荐)
        # 让每个基因受到的总调节量不至于过大
        # matrix = matrix / (np.linalg.norm(matrix, ord=1, axis=0, keepdims=True) + 1e-6)

        self.adj_matrix = matrix

    def add_trend_gene(self, edge_idx: int, n_genes: int, pattern='sigmoid', reverse=False, sticky=False, params: Optional[dict] = None, include_cycle_rim: bool = True):
        """
        添加趋势基因（sigmoid 或 impulse 模式）

        Args:
            edge_idx: Edge 索引
            n_genes: 基因数量
            pattern: 模式类型 ('sigmoid' 或 'impulse')
            reverse: 是否反转趋势
            sticky: 是否启用 sticky 模式
            params: 模式参数（center, slope/width）
            include_cycle_rim: 如果 Edge 连接到 CycleNode，是否将圆环(Rim)视为 Edge 的一部分
                              True: 视为圆环的一部分（细胞在圆环里运动时，表达依然活跃）
                              False: 不视为圆环的一部分（只计算到线段的距离）
        """
        indices = self._allocate_indices(n_genes)

        if params is None:
            params = {}
            if pattern == 'sigmoid':
                lo, hi = self.sigmoid_center_range
                params['center'] = np.random.rand(n_genes).astype(np.float32) * (hi - lo) + lo
                slo, shi = self.sigmoid_slope_range
                params['slope'] = np.random.rand(n_genes).astype(np.float32) * (shi - slo) + slo
            elif pattern == 'impulse':
                lo, hi = self.impulse_center_range
                params['center'] = np.random.rand(n_genes).astype(np.float32) * (hi - lo) + lo
                wlo, whi = self.impulse_width_range
                params['width'] = np.random.rand(n_genes).astype(np.float32) * (whi - wlo) + wlo

        self.modules.append({
            "kind": "edge",
            "type": pattern,
            "target_idx": edge_idx,
            "indices": indices,
            "params": params,
            "reverse": reverse,
            "sticky": sticky,
            "include_cycle_rim": include_cycle_rim
        })

    def add_cycle_gene(self, node_id: str, n_genes: int, phase: Optional[List[float]] = None, angular_sigma: float = 0.8):
        """
        添加Cycle基因

        Args:
            node_id: Cycle节点ID
            n_genes: 基因数量
            phase: 相位列表（如果None，则均匀分布在[0, 2π]）
            angular_sigma: 高斯峰的角宽度（单位：弧度），控制光斑的角宽度
                          - 0.3 ~ 0.5: 锐利（约17~28度），适合热图
                          - 0.8 ~ 1.0: 宽阔（约45~60度），适合PCA/UMAP建图
        """
        if phase is None:
            phases = np.linspace(0, 2 * np.pi, n_genes).astype(np.float32)
        else:
            phases = np.asarray(phase, dtype=np.float32)

        indices = self._allocate_indices(n_genes)

        self.modules.append({
            "kind": "cycle",
            "target_id": node_id,
            "indices": indices,
            "params": {
                "phases": phases,
                "angular_sigma": float(angular_sigma)
            }
        })

    def add_noise_gene(self, n_genes: int, alpha_range: tuple = (0.01, 0.05)):
        """
        添加噪声基因（白噪声类型）

        Args:
            n_genes: 基因数量
            alpha_range: alpha（转录速率）的范围，默认(0.01, 0.05)表示非常小的值
        """
        indices = self._allocate_indices(n_genes)

        # 为每个基因随机生成一个alpha值
        np.random.seed(self.seed if self.seed is not None else None)
        alpha_values = np.random.uniform(alpha_range[0], alpha_range[1], n_genes)

        self.modules.append({
            "kind": "noise",
            "indices": indices,
            "params": {
                "alpha_values": np.asarray(alpha_values, dtype=np.float32)
            }
        })

    def set_gene_kinetics(self, gene_indices: Union[int, List[int], np.ndarray],
                          beta: Optional[float] = None,
                          gamma: Optional[float] = None):
        """
        设置特定基因的动力学参数

        Args:
            gene_indices: 基因索引（可以是单个索引、列表或数组）
            beta: 剪接速率（splicing rate），如果为None则使用默认值
            gamma: 降解速率（degradation rate），如果为None则使用默认值
        """
        if isinstance(gene_indices, int):
            gene_indices = [gene_indices]
        elif isinstance(gene_indices, np.ndarray):
            gene_indices = gene_indices.tolist()

        for idx in gene_indices:
            if idx < 0 or idx >= self.n_genes:
                raise ValueError(f"Gene index {idx} out of range [0, {self.n_genes})")
            if beta is not None:
                self.beta[idx] = float(beta)
            if gamma is not None:
                self.gamma[idx] = float(gamma)

    # =====================================================
    # 3. 配置文件加载功能
    # =====================================================

    @staticmethod
    def _load_gene_config(config_path: str) -> List[Dict[str, Any]]:
        """
        从YAML文件加载基因配置（内部方法）

        Args:
            config_path: YAML配置文件路径

        Returns:
            基因配置列表
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config.get('genes', [])

    @staticmethod
    def get_config_info(config_path: str) -> Dict[str, Any]:
        """
        从配置文件获取基因信息摘要（静态方法）

        Args:
            config_path: YAML配置文件路径

        Returns:
            包含基因统计信息的字典
        """
        genes_config = GeneSynthesizer._load_gene_config(config_path)

        stats = {
            'total': len(genes_config),
            'by_type': {},
            'by_edge': {},
            'by_node': {},
            'sticky_count': 0
        }

        for gene_config in genes_config:
            gene_type = gene_config.get('type')
            stats['by_type'][gene_type] = stats['by_type'].get(gene_type, 0) + 1

            if gene_type == 'cycle':
                node_id = gene_config.get('node_id')
                stats['by_node'][node_id] = stats['by_node'].get(node_id, 0) + 1
            else:
                edge_idx = gene_config.get('edge_idx')
                if edge_idx is not None:
                    stats['by_edge'][edge_idx] = stats['by_edge'].get(edge_idx, 0) + 1

            if gene_config.get('sticky', False):
                stats['sticky_count'] += 1

        return stats

    def load_from_config(self, config_path: str, verbose: bool = True) -> Dict[str, int]:
        """
        从配置文件批量添加基因

        Args:
            config_path: YAML配置文件路径
            verbose: 是否打印详细信息

        Returns:
            基因名称到索引的映射字典
        """
        genes_config = self._load_gene_config(config_path)

        gene_name_to_idx = {}
        cycle_phases = {}  # 用于批量添加cycle基因
        noise_genes_list = []  # 用于批量添加noise基因

        # 第一遍：收集cycle基因的相位和noise基因
        for gene_config in genes_config:
            if gene_config.get('type') == 'cycle':
                node_id = gene_config.get('node_id')
                phase = gene_config.get('phase', 0.0)
                # 如果没有名称，暂时用占位符，后续会用实际索引生成
                name = gene_config.get('name', None)

                if node_id not in cycle_phases:
                    cycle_phases[node_id] = {'phases': [], 'names': [], 'configs': []}

                cycle_phases[node_id]['phases'].append(phase)
                cycle_phases[node_id]['names'].append(name)
                cycle_phases[node_id]['configs'].append(gene_config)
            elif gene_config.get('type') == 'noise':
                noise_genes_list.append(gene_config)

        # 批量添加cycle基因
        for node_id, data in cycle_phases.items():
            phases = data['phases']
            names = data['names']
            configs = data['configs']

            # 获取 angular_sigma（如果配置中有，使用第一个基因的配置；如果没有，使用默认值0.8）
            angular_sigma = 0.8  # 默认值
            if configs and 'angular_sigma' in configs[0]:
                angular_sigma = configs[0]['angular_sigma']

            # 添加cycle基因
            start_idx = self.current_gene_idx
            self.add_cycle_gene(node_id=node_id, n_genes=len(phases), phase=phases, angular_sigma=angular_sigma)

            # 记录基因名称到索引的映射，并设置动力学参数
            for i, (name, gene_config) in enumerate(zip(names, configs)):
                gene_idx = start_idx + i

                # 如果没有名称，用索引生成
                if name is None:
                    name = f'gene_{gene_idx}'

                # 存储基因名称
                self.gene_names[gene_idx] = name
                gene_name_to_idx[name] = gene_idx

                # 设置动力学参数
                beta = gene_config.get('beta')
                gamma = gene_config.get('gamma')
                if beta is not None or gamma is not None:
                    self.set_gene_kinetics(gene_idx, beta=beta, gamma=gamma)

            if verbose:
                final_names = [self.gene_names[start_idx + i] for i in range(len(phases))]
                sigma_info = f" [angular_sigma={angular_sigma:.2f}]" if angular_sigma != 0.8 else ""
                print(f"  ✓ Added {len(phases)} cycle genes to node '{node_id}': {final_names}{sigma_info}")

        # 批量添加noise基因
        if noise_genes_list:
            # 收集所有noise基因的alpha值
            alpha_values = []
            names = []
            configs = []
            for gene_config in noise_genes_list:
                alpha = gene_config.get('alpha', 0.03)  # 默认alpha值
                alpha_values.append(alpha)
                names.append(gene_config.get('name', None))
                configs.append(gene_config)

            # 计算alpha范围（基于所有alpha值的范围）
            if alpha_values:
                alpha_min = min(alpha_values) * 0.8
                alpha_max = max(alpha_values) * 1.2
                alpha_range = (alpha_min, alpha_max)
            else:
                alpha_range = (0.01, 0.05)

            start_idx = self.current_gene_idx
            self.add_noise_gene(
                n_genes=len(noise_genes_list),
                alpha_range=alpha_range
            )

            # 记录基因名称到索引的映射，并设置动力学参数
            for i, (name, gene_config) in enumerate(zip(names, configs)):
                gene_idx = start_idx + i

                # 如果没有名称，用索引生成
                if name is None:
                    name = f'noise_{gene_idx}'

                # 存储基因名称
                self.gene_names[gene_idx] = name
                gene_name_to_idx[name] = gene_idx

                # 设置动力学参数
                beta = gene_config.get('beta')
                gamma = gene_config.get('gamma')
                if beta is not None or gamma is not None:
                    self.set_gene_kinetics(gene_idx, beta=beta, gamma=gamma)

            if verbose:
                final_names = [self.gene_names[start_idx + i] for i in range(len(noise_genes_list))]
                print(f"  ✓ Added {len(noise_genes_list)} noise genes: {final_names[:5]}{'...' if len(final_names) > 5 else ''}")

        # 第二遍：添加其他类型的基因
        for gene_config in genes_config:
            gene_type = gene_config.get('type')

            if gene_type == 'cycle' or gene_type == 'noise':
                continue  # 已经在第一遍处理了

            if gene_type in ['sigmoid', 'impulse']:
                edge_idx = gene_config.get('edge_idx')
                if edge_idx is None:
                    # 临时名称用于错误信息
                    temp_name = gene_config.get('name', f'gene_{self.current_gene_idx}')
                    raise ValueError(f"Gene '{temp_name}': edge_idx is required for {gene_type} genes")

                sticky = gene_config.get('sticky', False)
                include_cycle_rim = gene_config.get('include_cycle_rim', True)  # 默认 True 保持向后兼容

                # reverse 只对 sigmoid 有效
                reverse = False
                if gene_type == 'sigmoid':
                    reverse = gene_config.get('reverse', False)
                elif gene_config.get('reverse', False):
                    # 如果 impulse 配置中设置了 reverse，给出警告但忽略
                    temp_name = gene_config.get('name', f'gene_{self.current_gene_idx}')
                    import warnings
                    warnings.warn(f"Gene '{temp_name}': reverse参数仅适用于sigmoid模式，impulse模式不支持reverse，已忽略")

                # 准备参数
                params = {}
                if gene_type == 'sigmoid':
                    if 'center' in gene_config:
                        params['center'] = np.asarray([gene_config['center']], dtype=np.float32)
                    if 'slope' in gene_config:
                        params['slope'] = np.asarray([gene_config['slope']], dtype=np.float32)
                elif gene_type == 'impulse':
                    if 'center' in gene_config:
                        params['center'] = np.asarray([gene_config['center']], dtype=np.float32)
                    if 'width' in gene_config:
                        params['width'] = np.asarray([gene_config['width']], dtype=np.float32)

                # 添加基因
                start_idx = self.current_gene_idx
                self.add_trend_gene(
                    edge_idx=edge_idx,
                    n_genes=1,
                    pattern=gene_type,
                    reverse=reverse,
                    sticky=sticky,
                    params=params if params else None,
                    include_cycle_rim=include_cycle_rim
                )

                # 如果没有名称，用索引生成
                name = gene_config.get('name')
                if name is None:
                    name = f'gene_{start_idx}'

                # 存储基因名称
                self.gene_names[start_idx] = name
                gene_name_to_idx[name] = start_idx

                # 设置动力学参数（如果配置中有）
                beta = gene_config.get('beta')
                gamma = gene_config.get('gamma')
                if beta is not None or gamma is not None:
                    self.set_gene_kinetics(start_idx, beta=beta, gamma=gamma)

                if verbose:
                    gene_type_str = f"{gene_type}{' (sticky)' if sticky else ''}"
                    kinetics_info = ""
                    if beta is not None or gamma is not None:
                        kinetics_info = f" [β={beta if beta is not None else self.default_beta:.2f}, γ={gamma if gamma is not None else self.default_gamma:.2f}]"
                    rim_info = f" [rim={'include' if include_cycle_rim else 'exclude'}]"
                    print(f"  ✓ Added {gene_type_str} gene '{name}' to edge {edge_idx}{kinetics_info}{rim_info}")
            else:
                raise ValueError(f"Unknown gene type: {gene_type}")

        if verbose:
            print(f"\n✓ Total genes added: {len(gene_name_to_idx)}")
            print(f"  Gene name to index mapping:")
            for name, idx in sorted(gene_name_to_idx.items(), key=lambda x: x[1]):
                print(f"    {name}: Gene {idx}")

        return gene_name_to_idx

    # =====================================================
    # 2. 核心计算逻辑 (Refactored)
    # =====================================================

    def _compute_distance_to_cycle_rim(self, z, cycle_node):
        """辅助函数：计算点到 CycleNode 圆环的距离"""
        center = np.asarray(cycle_node.pos, dtype=z.dtype)
        diff = z - center
        dist_to_center = np.linalg.norm(diff, axis=1)
        # 到圆环(Rim)的距离 = |距离中心的距离 - 半径|
        dist_to_rim = np.abs(dist_to_center - cycle_node.radius)
        return dist_to_rim

    def _compute_transcription_rate(self, z: np.ndarray, current_s: Optional[np.ndarray] = None, scale=1.5) -> np.ndarray:
        """
        核心修正：计算转录速率，包含对 CycleNode 边界的几何修正
        """
        N = z.shape[0]
        alpha = np.zeros((N, self.n_genes), dtype=np.float32)

        for mod in self.modules:
            indices = mod['indices']

            if mod['kind'] == 'edge':
                edge_idx = mod['target_idx']
                edge = self.system.edges[edge_idx]

                # --- 1. 计算几何投影 (t) ---
                t = edge.get_projection_t(z)[:, None]
                t = np.clip(t, 0.0, 1.0)
                if mod['reverse']: t = 1.0 - t

                # --- 2. 空间门控 (Spatial Gating) [关键修正] ---
                # 基础距离：到线段的距离
                d2 = edge.distance_sq(z)[:, None]

                # [Fix] 根据基因类型决定是否将 CycleNode 的圆环视为 Edge 的一部分
                # include_cycle_rim=True: 把圆环(Rim)视为 Edge 的一部分
                #   这样细胞在圆环里运动时，d2 依然接近 0，spatial_gate 依然接近 1
                # include_cycle_rim=False: 只计算到线段的距离，不考虑圆环
                include_rim = mod.get('include_cycle_rim', True)  # 默认 True 保持向后兼容
                if include_rim:
                    if isinstance(edge.node_start, CycleNode):
                        dist_rim_start = self._compute_distance_to_cycle_rim(z, edge.node_start)[:, None]
                        d2 = np.minimum(d2, dist_rim_start ** 2)

                    if isinstance(edge.node_end, CycleNode):
                        dist_rim_end = self._compute_distance_to_cycle_rim(z, edge.node_end)[:, None]
                        d2 = np.minimum(d2, dist_rim_end ** 2)

                gate_width = edge.sigma_start * scale
                spatial_gate = np.exp(-d2 / (2 * gate_width**2))

                # --- 3. 模式计算 (Pattern) ---
                if mod['type'] == 'sigmoid':
                    c = mod['params']['center']
                    s = mod['params']['slope']
                    expr_local = _sigmoid(s * (t - c))
                elif mod['type'] == 'impulse':
                    c = mod['params']['center']
                    w = mod['params']['width']
                    expr_local = np.exp(-((t - c)**2) / (2 * w**2))
                else:
                    expr_local = np.zeros_like(t)

                total_expr = expr_local * spatial_gate

                # --- 4. Sticky 逻辑 ---
                # 现在的 Sticky 逻辑会更健壮，因为当前 Edge 在 Cycle 里本身就亮着
                if mod.get('sticky', False) and not mod['reverse']:
                    downstream_indices = self.system.get_downstream_edges(edge_idx)
                    if downstream_indices:
                        downstream_gate_sum = np.zeros_like(spatial_gate)
                        for d_idx in downstream_indices:
                            d_edge = self.system.edges[d_idx]

                            # 对下游 Edge 也应用同样的 Cycle 修正逻辑（使用相同的 include_cycle_rim 设置）
                            d_d2 = d_edge.distance_sq(z)[:, None]

                            if include_rim:
                                if isinstance(d_edge.node_start, CycleNode):
                                    d_dist_rim_s = self._compute_distance_to_cycle_rim(z, d_edge.node_start)[:, None]
                                    d_d2 = np.minimum(d_d2, d_dist_rim_s ** 2)
                                if isinstance(d_edge.node_end, CycleNode):
                                    d_dist_rim_e = self._compute_distance_to_cycle_rim(z, d_edge.node_end)[:, None]
                                    d_d2 = np.minimum(d_d2, d_dist_rim_e ** 2)

                            d_gate_width = d_edge.sigma_start * scale
                            d_gate = np.exp(-d_d2 / (2 * d_gate_width**2))
                            downstream_gate_sum = np.maximum(downstream_gate_sum, d_gate)

                        total_expr = np.maximum(total_expr, downstream_gate_sum)

                alpha[:, indices] += total_expr * self.edge_strength

            elif mod['kind'] == 'cycle':
                # Cycle 基因逻辑保持不变
                try:
                    c_node = next(n for n in self.system.nodes if n.id == mod['target_id'])
                except StopIteration:
                    continue

                center = np.asarray(c_node.pos, dtype=z.dtype)
                diff = z - center
                angle = np.arctan2(diff[:, 1], diff[:, 0])[:, None]
                angle = np.where(angle < 0, angle + 2*np.pi, angle)

                dist_rim = (np.linalg.norm(diff, axis=1) - c_node.radius)[:, None]
                cycle_scale_factor = 0.4
                gate_width = c_node.sigma * scale*cycle_scale_factor
                spatial_gate = np.exp(-(dist_rim**4) / (2 * gate_width**4))

                phases = mod['params']['phases']

                # A. 计算最小角距离 (Shortest Angular Distance)
                # 逻辑：将差值映射到 [-pi, pi] 区间
                angle_diff = angle - phases
                angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

                # B. 高斯峰参数
                # angular_sigma 控制光斑的角宽度 (单位：弧度)
                # 推荐值:
                #   0.3 ~ 0.5 (锐利，约 17~28度，适合热图)
                #   0.8 ~ 1.0 (宽阔，约 45~60度，适合 PCA/UMAP 建图)
                # 从配置中读取，如果没有则使用默认值 0.8
                angular_sigma = mod['params'].get('angular_sigma', 0.8)

                # C. 高斯公式
                expr = np.exp(-(angle_diff**2) / (2 * angular_sigma**2))

                alpha[:, indices] += expr * spatial_gate * self.cycle_strength

            elif mod['kind'] == 'noise':
                # 噪声基因：使用非常小的随机alpha值（白噪声）
                alpha_values = mod['params']['alpha_values']  # [n_noise_genes]
                # 为每个细胞添加小的随机波动
                N = z.shape[0]
                n_noise_genes = len(indices)
                # 确保alpha_values的长度与indices匹配
                if len(alpha_values) != n_noise_genes:
                    raise ValueError(f"alpha_values长度({len(alpha_values)})与indices长度({n_noise_genes})不匹配")
                # 生成随机噪声矩阵 [N, n_noise_genes]
                if self.seed is not None:
                    np.random.seed(int(self.seed + mod['indices'][0]))  # 使用不同的种子确保可重复性
                noise = np.random.rand(N, n_noise_genes).astype(np.float32) * 0.1  # 10%的随机波动
                # alpha = base_alpha + noise，确保alpha值很小
                alpha[:, indices] += (alpha_values[None, :] + noise)

        if self.seed is not None:
            np.random.seed(self.seed)

        if self.adj_matrix is not None and current_s is not None:
            # current_s: [Batch, n_genes]
            # adj_matrix: [n_genes, n_genes] (行j 调控 列i)

            # 1. 计算调控输入 (Matrix Multiplication)
            # result: [Batch, n_genes]
            regulation_input = np.matmul(current_s, self.adj_matrix)

            # 2. 应用非线性激活函数 (tanh)
            # 将调节限制在 [-1, 1] 之间，防止数值爆炸
            # 如果 input > 0 (激活), tanh -> 1
            # 如果 input < 0 (抑制), tanh -> -1
            reg_effect = np.tanh(regulation_input)

            # 3. 叠加到几何 Alpha 上
            # alpha_new = alpha_geo + strength * effect
            alpha += self.grn_strength * reg_effect

        # ========================================================
        # [新增逻辑]：全局动力学衰减 (Global Kinetic Decay)
        # 模拟分化过程中转录速率的整体下降 (Neuroblast -> Granule)
        # 生物学事实：在神经发生过程中，从干细胞/前体细胞（高转录）到成熟神经元（低转录）
        # ========================================================
        if self.enable_global_decay and len(self.system.edges) > 0:
            # 1. 计算每个细胞在主 Edge 上的投影进度 t (0.0 -> 1.0)
            # 使用 Edge 0 作为主干道（Neuroblast -> Granule）
            main_edge = self.system.edges[0]
            t_proj = main_edge.get_projection_t(z)  # [N]
            t_proj = np.clip(t_proj, 0.0, 1.0)[:, None]  # [N, 1]

            # 2. 定义衰减函数
            # t=0 (Neuroblast): 因子 = start_rate (高转录，约 1.3x)
            # t=1 (Granule):    因子 = end_rate (低转录，约 0.6x)
            # 线性衰减公式: factor = start_rate - (start_rate - end_rate) * t
            decay_factor = self.global_decay_start_rate - (self.global_decay_start_rate - self.global_decay_end_rate) * t_proj  # [N, 1]

            # 3. 应用到所有基因（全局衰减）
            # 使用广播：decay_factor [N, 1] 应用到 alpha [N, n_genes]
            alpha = alpha * decay_factor

        # ========================================================

        alpha += np.random.rand(*alpha.shape).astype(np.float32) * float(self.background_noise)
        alpha = np.clip(alpha, 0.0, None)
        self.alpha = alpha
        return alpha

    def _draw_capture_factor(self, expr_rate: np.ndarray) -> np.ndarray:
        """Draw a cell-specific capture factor for absolute-abundance counts."""
        N = expr_rate.shape[0]
        lib_size = np.random.lognormal(mean=np.log(self.lib_size_mean), sigma=self.lib_size_sd, size=(N, 1)).astype(np.float32)
        total_abundance = np.clip(expr_rate.sum(axis=1, keepdims=True), 1e-6, None)
        reference_total = np.clip(np.median(total_abundance), 1e-6, None)
        return lib_size / reference_total

    def _sample_counts(
        self,
        expr_rate: np.ndarray,
        capture_factor: Optional[np.ndarray] = None,
        capture_scale: float = 1.0,
    ) -> np.ndarray:
        """Sample counts from absolute abundances with a cell-specific capture factor."""
        if capture_factor is None:
            capture_factor = self._draw_capture_factor(expr_rate)

        mean_counts = np.clip(expr_rate, 0.0, None) * capture_factor * float(capture_scale)

        # 采样计数
        if self.sampling == 'poisson':
            counts = np.random.poisson(mean_counts).astype(np.float32)
        elif self.sampling in ['nb', 'negative_binomial']:
            r = self.nb_dispersion
            mean_counts_clamped = np.clip(mean_counts, 1e-6, None)
            # torch Gamma(concentration=r, rate=r/mean)  <->  numpy gamma(shape=r, scale=mean/r)
            lam = np.random.gamma(shape=r, scale=mean_counts_clamped / r)
            counts = np.random.poisson(lam).astype(np.float32)
        else:
            counts = mean_counts # fallback

        # 应用dropout噪声（如果启用）
        if self.dropout_prob > 0:
            # Dropout概率与表达量相关：低表达基因更容易dropout
            # p_dropout = base_prob * exp(-slope * normalized_expression)
            # 归一化表达量（相对于每个细胞的平均表达）
            normalized_expr = mean_counts / (mean_counts.mean(axis=1, keepdims=True) + 1e-6)
            # 计算dropout概率：低表达 = 高dropout概率
            p_dropout = self.dropout_prob * np.exp(-self.dropout_slope * normalized_expr)
            # 随机决定哪些基因-细胞对发生dropout
            dropout_mask = np.random.rand(*counts.shape) < p_dropout
            counts[dropout_mask] = 0

        return counts

    # =====================================================
    # 3. 两种生成模式：静态 (Legacy) vs 动力学 (New)
    # =====================================================

    def generate(self, z_input: Union[np.ndarray, List], scale=1.5, simulate_counts=True):
        """
        [Deprecated] 为了向后兼容保留，内部调用 generate_kinetics

        注意：此方法已废弃，建议直接使用 generate_kinetics 方法。
        对于单时间点输入，此方法会调用 generate_kinetics 并返回 spliced 表达量。
        """
        import warnings
        warnings.warn(
            "generate() method is deprecated. Use generate_kinetics() instead. "
            "This method will call generate_kinetics() internally.",
            DeprecationWarning,
            stacklevel=2
        )

        # 调用 generate_kinetics，使用默认 dt
        result = self.generate_kinetics(
            traj_input=z_input,
            dt=0.01,
            scale=scale,
            simulate_counts=simulate_counts
        )

        # 返回 spliced 表达量（保持向后兼容）
        return result['spliced']

    def generate_kinetics(
        self,
        traj_input: Union[np.ndarray, List],
        dt: float = 0.01,
        scale: float = 1.5,
        simulate_counts: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        [Kinetic Mode] 动力学生成 (支持 RNA Velocity)

        现在统一使用动力学方法生成所有数据，包括单时间点的情况。

        Args:
            traj_input:
                - 3D数组 [Steps, Cells, Dims]: 完整轨迹，进行ODE积分
                - 2D数组 [Cells, Dims]: 单时间点，创建单步轨迹进行积分
            dt: 仿真时间步长 (必须与 GraphSimulator.run 中的 dt 一致)
            scale: 空间门控的缩放因子
            simulate_counts: 是否生成计数数据（带dropout噪声）

        Returns:
            Dict containing:
            - 'spliced': [N_total, Genes] 或 [Cells, Genes] (单时间点)
            - 'unspliced': [N_total, Genes] 或 [Cells, Genes] (单时间点)
            - 'velocity': [N_total, Genes] 或 [Cells, Genes] (Ground Truth Velocity)
            - 'alpha': [N_total, Genes] 或 [Cells, Genes] (Transcription Rate)
        """
        # 1. 处理输入
        if isinstance(traj_input, np.ndarray):
            traj = np.asarray(traj_input, dtype=np.float32)
        else:
            traj = np.asarray(traj_input, dtype=np.float32).copy()

        # 处理2D输入（单时间点）：转换为3D [1, Cells, Dims]
        if traj.ndim == 2:
            traj = traj[None, :, :]  # [1, Cells, Dims]
        elif traj.ndim == 3:
            pass  # 已经是3D，保持不变
        else:
            raise ValueError(f"generate_kinetics requires 2D [Cells, Dims] or 3D [Steps, Cells, Dims] array, got {traj.ndim}D")

        steps, n_cells, dims = traj.shape

        # 2. 初始化状态 (u: unspliced, s: spliced)
        # 设置稳态初始值（推荐，最物理）：在模拟的第一步，直接把细胞的 u 和 s 设置为起始基因表达状态的理论稳态值
        z_initial = traj[0]  # [Cells, Dims] - 初始位置

        # 计算初始位置的转录速率 alpha_0
        # 对于初始状态，current_s 可以设为 None 或零（因为还没有 spliced RNA）
        alpha_0 = self._compute_transcription_rate(z_initial, current_s=None, scale=scale)  # [Cells, Genes]

        # 获取基因级的 beta 和 gamma
        beta_t = self.beta[None, :]  # [1, n_genes] 用于广播
        gamma_t = self.gamma[None, :]  # [1, n_genes] 用于广播

        # 计算理论稳态值：
        # du/dt = 0 => alpha - beta * u = 0 => u_steady = alpha / beta
        # ds/dt = 0 => beta * u - gamma * s = 0 => s_steady = beta * u / gamma = alpha / gamma
        # 避免除零，添加小的 epsilon
        epsilon = 1e-8
        u = alpha_0 / (beta_t + epsilon)  # [Cells, Genes]
        s = alpha_0 / (gamma_t + epsilon)  # [Cells, Genes]

        # 确保非负（虽然理论上应该已经非负）
        u = np.clip(u, 0.0, None)
        s = np.clip(s, 0.0, None)

        # 存储历史记录 (如果只需要最终快照，可以只存最后，但为了生成完整数据集，通常保留所有点)
        u_history = []
        s_history = []
        alpha_history = []
        vel_history = [] # ds/dt

        if steps > 1:
            print(f"Integrating gene kinetics over {steps} steps (starting from steady state)...")
        else:
            print(f"Integrating gene kinetics for single time point (starting from steady state)...")

        # 3. 欧拉积分循环
        for i in range(steps):
            z_t = traj[i] # [Cells, Dims]

            # A. 计算当前的转录速率 alpha(t)
            alpha_t = self._compute_transcription_rate(z_t, current_s=s, scale=scale)

            # B. ODE 更新
            # du/dt = alpha - beta * u
            # ds/dt = beta * u - gamma * s
            # 使用基因级的 beta 和 gamma（逐元素运算）

            du = alpha_t - beta_t * u
            ds = beta_t * u - gamma_t * s

            # Euler step
            u = u + du * dt
            s = s + ds * dt

            # 物理约束：分子数不能为负
            u = np.clip(u, 0.0, None)
            s = np.clip(s, 0.0, None)

            # C. 记录数据
            u_history.append(u.copy())
            s_history.append(s.copy())
            alpha_history.append(alpha_t.copy())
            vel_history.append(ds.copy()) # 记录真实的 RNA 速度

        # 4. 处理输出格式
        # 如果是单时间点（steps=1），直接返回 [Cells, Genes]
        # 如果是多时间点，展平为 [Steps * Cells, Genes]
        if steps == 1:
            u_final = u_history[0]  # [Cells, Genes]
            s_final = s_history[0]  # [Cells, Genes]
            alpha_final = alpha_history[0]  # [Cells, Genes]
            vel_final = vel_history[0]  # [Cells, Genes]
        else:
            # 堆叠并展平 [Steps * Cells, Genes]
            # 这样格式可以兼容 AnnData 的 (n_obs, n_vars)
            u_final = np.stack(u_history).reshape(-1, self.n_genes)
            s_final = np.stack(s_history).reshape(-1, self.n_genes)
            alpha_final = np.stack(alpha_history).reshape(-1, self.n_genes)
            vel_final = np.stack(vel_history).reshape(-1, self.n_genes)

        # 保存无噪声版本
        self.expression_clean = s_final
        self.velocity_clean = vel_final

        # 5. 模拟计数 (可选)
        if simulate_counts:
            # Spliced and unspliced counts share the same cell-specific capture
            # factor; unspliced molecules use a fixed lower capture efficiency.
            capture_factor = self._draw_capture_factor(s_final)
            s_counts = self._sample_counts(s_final, capture_factor=capture_factor)
            u_counts = self._sample_counts(
                u_final,
                capture_factor=capture_factor,
                capture_scale=self.unspliced_capture_ratio,
            )

            return {
                'spliced': s_counts,
                'unspliced': u_counts,
                'velocity': vel_final, # GT RNA-state velocity is a continuous rate
                'alpha': alpha_final
            }
        else:
            return {
                'spliced': s_final,
                'unspliced': u_final,
                'velocity': vel_final,
                'alpha': alpha_final
            }
# Final
