#!/usr/bin/env python3
"""
从 UniversalGraphSystem 对象自动生成基因配置文件

核心功能：给定一个 system 对象和参数，输出基因配置字典。

用法:
    from generate_gene_config import auto_generate_config_from_system
    from graph_core import UniversalGraphSystem, GraphNode, CycleNode, GraphEdge
    
    # 创建系统
    system = UniversalGraphSystem(nodes, edges)
    
    # 生成配置
    config = auto_generate_config_from_system(
        system,
        n_genes_per_edge=30,
        n_genes_per_cycle=50,
        seed=42
    )
    
    # 保存配置（可选）
    import yaml
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
"""

import numpy as np
from typing import Dict, List, Any, Optional
import random

# 尝试导入系统相关类
try:
    from .graph_core import UniversalGraphSystem, CycleNode, GraphEdge
    HAS_SYSTEM_MODULE = True
except ImportError:
    try:
        from graph_core import UniversalGraphSystem, CycleNode, GraphEdge
        HAS_SYSTEM_MODULE = True
    except ImportError:
        HAS_SYSTEM_MODULE = False


class GeneConfigGenerator:
    """基因配置生成器"""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Args:
            seed: 随机种子
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.seed = seed
    
    def generate_cycle_genes(
        self,
        node_id: str,
        n_genes: int,
        phases: Optional[List[float]] = None,
        beta_range: Optional[tuple] = None,
        gamma_range: Optional[tuple] = None,
        angular_sigma: Optional[float] = None,
        name_prefix: str = "cycle"
    ) -> List[Dict[str, Any]]:
        """
        生成Cycle基因配置
        
        Args:
            node_id: Cycle节点ID
            n_genes: 基因数量
            phases: 相位列表（如果None，则均匀分布在[0, 2π]）
            beta_range: (min, max) beta范围，如果None则使用默认值
            gamma_range: (min, max) gamma范围，如果None则使用默认值
            angular_sigma: 高斯峰的角宽度（单位：弧度），如果None则使用默认值0.8
                          - 0.3 ~ 0.5: 锐利（约17~28度），适合热图
                          - 0.8 ~ 1.0: 宽阔（约45~60度），适合PCA/UMAP建图
            name_prefix: 基因名称前缀
            
        Returns:
            基因配置列表
        """
        genes = []
        
        if phases is None:
            phases = np.linspace(0, 2 * np.pi, n_genes).tolist()
        elif len(phases) != n_genes:
            raise ValueError(f"phases长度({len(phases)})必须等于n_genes({n_genes})")
        
        # 如果指定了angular_sigma，只在第一个基因中保存（所有基因共享同一个angular_sigma）
        for i, phase in enumerate(phases):
            gene_config = {
                'type': 'cycle',
                'node_id': node_id,
                'phase': float(phase),
                'name': f'{name_prefix}_{node_id}_{i}'
            }
            
            # 只在第一个基因中保存angular_sigma（所有基因共享）
            if angular_sigma is not None and i == 0:
                gene_config['angular_sigma'] = float(angular_sigma)
            
            # 随机生成beta和gamma（如果指定了范围）
            if beta_range is not None:
                gene_config['beta'] = float(np.random.uniform(beta_range[0], beta_range[1]))
            if gamma_range is not None:
                gene_config['gamma'] = float(np.random.uniform(gamma_range[0], gamma_range[1]))
            
            genes.append(gene_config)
        
        return genes
    
    def generate_noise_genes(
        self,
        n_genes: int,
        alpha_range: tuple = (0.01, 0.05),
        beta_range: Optional[tuple] = None,
        gamma_range: Optional[tuple] = None,
        name_prefix: str = "noise"
    ) -> List[Dict[str, Any]]:
        """
        生成噪声基因配置（白噪声类型）
        
        Args:
            n_genes: 基因数量
            alpha_range: alpha（转录速率）的范围，默认(0.01, 0.05)表示非常小的值
            beta_range: (min, max) beta范围，如果None则使用默认值
            gamma_range: (min, max) gamma范围，如果None则使用默认值
            name_prefix: 基因名称前缀
            
        Returns:
            基因配置列表
        """
        genes = []
        
        for i in range(n_genes):
            gene_config = {
                'type': 'noise',
                'name': f'{name_prefix}_{i}',
                'alpha': float(np.random.uniform(alpha_range[0], alpha_range[1]))
            }
            
            # 随机生成beta和gamma（如果指定了范围）
            if beta_range is not None:
                gene_config['beta'] = float(np.random.uniform(beta_range[0], beta_range[1]))
            if gamma_range is not None:
                gene_config['gamma'] = float(np.random.uniform(gamma_range[0], gamma_range[1]))
            
            genes.append(gene_config)
        
        return genes
    
    def generate_trend_genes(
        self,
        edge_idx: int,
        n_genes: int,
        pattern: str = 'sigmoid',  # 'sigmoid' or 'impulse'
        reverse: bool = False,
        sticky: bool = False,
        center_range: Optional[tuple] = None,
        slope_range: Optional[tuple] = None,  # 仅用于sigmoid
        width_range: Optional[tuple] = None,  # 仅用于impulse
        beta_range: Optional[tuple] = None,
        gamma_range: Optional[tuple] = None,
        name_prefix: Optional[str] = None,
        include_cycle_rim: bool = True,
        # 默认参数范围（当center_range等为None时使用）
        default_center_range: Optional[tuple] = None,
        default_slope_range: Optional[tuple] = None,
        default_width_range: Optional[tuple] = None,
    ) -> List[Dict[str, Any]]:
        """
        生成Trend基因配置（sigmoid或impulse）
        
        Args:
            edge_idx: Edge索引
            n_genes: 基因数量
            pattern: 'sigmoid' 或 'impulse'
            reverse: 是否反向（仅用于sigmoid，表示一开始高表达，后来低表达）
            sticky: 是否sticky
            center_range: (min, max) center范围
            slope_range: (min, max) slope范围（仅sigmoid）
            width_range: (min, max) width范围（仅impulse）
            beta_range: (min, max) beta范围
            gamma_range: (min, max) gamma范围
            name_prefix: 基因名称前缀（如果None，自动生成）
            include_cycle_rim: 如果 Edge 连接到 CycleNode，是否将圆环(Rim)视为 Edge 的一部分
            default_center_range: 默认center范围（当center_range为None时使用）
            default_slope_range: 默认slope范围（当slope_range为None时使用）
            default_width_range: 默认width范围（当width_range为None时使用）
            
        Returns:
            基因配置列表
        """
        if pattern not in ['sigmoid', 'impulse']:
            raise ValueError(f"pattern必须是'sigmoid'或'impulse'，得到'{pattern}'")
        
        # reverse 只对 sigmoid 有效
        if pattern == 'impulse' and reverse:
            raise ValueError("reverse参数仅适用于sigmoid模式，impulse模式不支持reverse")
        
        if name_prefix is None:
            name_prefix = f'{pattern}_edge{edge_idx}'
        
        # 使用默认范围（如果提供了）
        if center_range is None and default_center_range is not None:
            center_range = default_center_range
        if pattern == 'sigmoid' and slope_range is None and default_slope_range is not None:
            slope_range = default_slope_range
        elif pattern == 'impulse' and width_range is None and default_width_range is not None:
            width_range = default_width_range
        
        genes = []
        
        for i in range(n_genes):
            gene_config = {
                'type': pattern,
                'edge_idx': edge_idx,
                'name': f'{name_prefix}_{i}'
            }
            
            # reverse 只对 sigmoid 有效
            if pattern == 'sigmoid' and reverse:
                gene_config['reverse'] = True
            if sticky:
                gene_config['sticky'] = True
            
            # 添加 include_cycle_rim 参数
            gene_config['include_cycle_rim'] = include_cycle_rim
            
            # 生成参数（总是生成，如果提供了范围）
            if center_range is not None:
                gene_config['center'] = float(np.random.uniform(center_range[0], center_range[1]))
            
            if pattern == 'sigmoid':
                if slope_range is not None:
                    gene_config['slope'] = float(np.random.uniform(slope_range[0], slope_range[1]))
            elif pattern == 'impulse':
                if width_range is not None:
                    gene_config['width'] = float(np.random.uniform(width_range[0], width_range[1]))
            
            # 随机生成beta和gamma
            if beta_range is not None:
                gene_config['beta'] = float(np.random.uniform(beta_range[0], beta_range[1]))
            if gamma_range is not None:
                gene_config['gamma'] = float(np.random.uniform(gamma_range[0], gamma_range[1]))
            
            genes.append(gene_config)
        
        return genes
    
    def generate_config(
        self,
        # Cycle基因配置
        cycle_genes: Optional[Dict[str, int]] = None,  # {node_id: n_genes}
        cycle_phases: Optional[Dict[str, List[float]]] = None,  # {node_id: [phases]}
        cycle_beta_range: Optional[tuple] = None,
        cycle_gamma_range: Optional[tuple] = None,
        cycle_angular_sigma: Optional[float] = None,  # Cycle基因的angular_sigma（高斯峰角宽度）
        cycle_name_prefix: str = "cycle",
        
        # Edge基因配置
        edge_genes: Optional[Dict[int, Dict[str, Any]]] = None,  # {edge_idx: {n_genes, pattern, ...}}
        edge_beta_range: Optional[tuple] = None,
        edge_gamma_range: Optional[tuple] = None,
        
        # 噪声基因配置
        noise_genes: Optional[int] = None,  # 噪声基因数量
        noise_alpha_range: tuple = (0.01, 0.05),  # 噪声基因的alpha范围
        noise_beta_range: Optional[tuple] = None,
        noise_gamma_range: Optional[tuple] = None,
        noise_name_prefix: str = "noise",
        
        # 默认参数范围（如果未指定）
        default_sigmoid_center_range: tuple = (0.2, 0.8),
        default_sigmoid_slope_range: tuple = (5.0, 15.0),
        default_impulse_center_range: tuple = (0.1, 0.9),
        default_impulse_width_range: tuple = (0.05, 0.15),
    ) -> Dict[str, Any]:
        """
        生成完整的基因配置
        
        Args:
            cycle_genes: {node_id: n_genes} 字典
            cycle_phases: {node_id: [phases]} 字典（可选）
            cycle_beta_range: Cycle基因的beta范围
            cycle_gamma_range: Cycle基因的gamma范围
            cycle_angular_sigma: Cycle基因的angular_sigma（高斯峰角宽度，单位：弧度）
                                 - 0.3 ~ 0.5: 锐利（约17~28度），适合热图
                                 - 0.8 ~ 1.0: 宽阔（约45~60度），适合PCA/UMAP建图
                                 - 如果None，使用默认值0.8
            cycle_name_prefix: Cycle基因名称前缀
            
            edge_genes: {edge_idx: {n_genes, pattern, reverse, sticky, ...}} 字典
            edge_beta_range: Edge基因的beta范围
            edge_gamma_range: Edge基因的gamma范围
            
            默认参数范围...
            
        Returns:
            完整的配置字典 {'genes': [...]}
        """
        all_genes = []
        
        # 生成Cycle基因
        if cycle_genes:
            for node_id, n_genes in cycle_genes.items():
                phases = cycle_phases.get(node_id) if cycle_phases else None
                genes = self.generate_cycle_genes(
                    node_id=node_id,
                    n_genes=n_genes,
                    phases=phases,
                    beta_range=cycle_beta_range,
                    gamma_range=cycle_gamma_range,
                    angular_sigma=cycle_angular_sigma,
                    name_prefix=cycle_name_prefix
                )
                all_genes.extend(genes)
        
        # 生成Edge基因
        if edge_genes:
            for edge_idx, config in edge_genes.items():
                # 支持两种格式：
                # 1. 简单格式：单个配置字典
                # 2. 列表格式：多个配置（同一edge多种模式）
                if isinstance(config, list):
                    configs = config
                else:
                    configs = [config]
                
                for cfg in configs:
                    n_genes = cfg.get('n_genes', 1)
                    pattern = cfg.get('pattern', 'sigmoid')
                    sticky = cfg.get('sticky', False)
                    include_cycle_rim = cfg.get('include_cycle_rim', True)  # 默认 True
                    
                    # reverse 只对 sigmoid 有效
                    reverse = False
                    if pattern == 'sigmoid':
                        reverse = cfg.get('reverse', False)
                    elif cfg.get('reverse', False):
                        # 如果 impulse 配置中设置了 reverse，给出警告但忽略
                        import warnings
                        warnings.warn(f"Edge {edge_idx} 的 impulse 模式不支持 reverse 参数，已忽略")
                    
                    # 获取参数范围
                    if pattern == 'sigmoid':
                        center_range = cfg.get('center_range', default_sigmoid_center_range)
                        slope_range = cfg.get('slope_range', default_sigmoid_slope_range)
                        width_range = None
                        default_slope_range = default_sigmoid_slope_range
                        default_width_range = None
                    else:  # impulse
                        center_range = cfg.get('center_range', default_impulse_center_range)
                        width_range = cfg.get('width_range', default_impulse_width_range)
                        slope_range = None
                        default_slope_range = None
                        default_width_range = default_impulse_width_range
                    
                    genes = self.generate_trend_genes(
                        edge_idx=edge_idx,
                        n_genes=n_genes,
                        pattern=pattern,
                        reverse=reverse,
                        sticky=sticky,
                        center_range=center_range,
                        slope_range=slope_range,
                        width_range=width_range,
                        beta_range=edge_beta_range,
                        gamma_range=edge_gamma_range,
                        include_cycle_rim=include_cycle_rim,
                        default_center_range=default_sigmoid_center_range if pattern == 'sigmoid' else default_impulse_center_range,
                        default_slope_range=default_slope_range,
                        default_width_range=default_width_range
                    )
                    all_genes.extend(genes)
        
        # 生成噪声基因（noise类型）
        if noise_genes is not None and noise_genes > 0:
            genes = self.generate_noise_genes(
                n_genes=noise_genes,
                alpha_range=noise_alpha_range,
                beta_range=noise_beta_range,
                gamma_range=noise_gamma_range,
                name_prefix=noise_name_prefix
            )
            all_genes.extend(genes)
        
        return {'genes': all_genes}


def auto_generate_config_from_system(
    system,
    n_genes_per_edge: int = 30,
    n_genes_per_cycle: int = 50,
    seed: int = 42,
    # 动力学参数范围
    cycle_beta_range: tuple = (0.8, 1.2),
    cycle_gamma_range: tuple = (0.4, 0.6),
    cycle_angular_sigma: Optional[float] = 0.8,  # Cycle基因的angular_sigma（高斯峰角宽度）
    edge_beta_range: tuple = (0.5, 1.5),
    edge_gamma_range: tuple = (0.4, 0.6),
    # 参数范围
    sigmoid_center_range: tuple = (0.2, 0.8),
    sigmoid_slope_range: tuple = (5.0, 15.0),
    impulse_center_range: tuple = (0.1, 0.9),
    impulse_width_range: tuple = (0.05, 0.15),
    # Edge基因分布配置（优先级最高）
    edge_gene_distribution: Optional[Dict[int, Dict[str, int]]] = None,
    # 格式: {edge_idx: {'sigmoid': 30, 'sigmoid_reverse': 10, 'sticky': 20, 'impulse': 20}}
    # - 'sigmoid': 正向sigmoid基因（sticky=False）
    # - 'sigmoid_reverse': 反向sigmoid基因（sticky=False）
    # - 'sticky': 正向sigmoid基因（sticky=True，独立类型）
    # - 'impulse': impulse基因
    # 如果提供了此参数，将忽略下面的自动分配策略参数
    
    # 自动分配策略（仅在 edge_gene_distribution 为 None 时使用）
    first_edge_has_background: bool = True,  # Edge 0 是否包含背景噪音基因
    later_edges_sticky: bool = True,  # 后续Edge是否使用sticky模式
    # Edge模式配置
    default_edge_pattern: str = 'sigmoid',  # 默认Edge模式：'sigmoid' 或 'impulse'
    edge_pattern: Optional[Dict[int, str]] = None,  # {edge_idx: 'sigmoid'|'impulse'} 为每个Edge指定模式（覆盖默认值）
    # 噪声基因配置
    n_noise_genes: int = 30,  # 噪声基因数量（noise类型，白噪声，不依赖于edge）
    noise_alpha_range: tuple = (0.01, 0.05),  # 噪声基因的alpha范围（转录速率，非常小的值）
    noise_beta_range: Optional[tuple] = None,  # 噪声基因的beta范围（如果None，使用edge_beta_range）
    noise_gamma_range: Optional[tuple] = None,  # 噪声基因的gamma范围（如果None，使用edge_gamma_range）
    # Cycle Rim 设置
    default_include_cycle_rim: bool = True,  # 默认是否将圆环视为Edge的一部分
    edge_include_cycle_rim: Optional[Dict[int, bool]] = None,  # 每个Edge的include_cycle_rim设置（覆盖默认值）
) -> Dict[str, Any]:
    """
    从 UniversalGraphSystem 对象自动生成基因配置文件
    
    这是本模块的核心函数：给定一个 system 对象和参数，输出基因配置字典。
    
    Args:
        system: UniversalGraphSystem 对象
        n_genes_per_edge: 每个Edge的基因数量
        n_genes_per_cycle: 每个Cycle节点的基因数量
        seed: 随机种子
        cycle_beta_range: Cycle基因的beta范围
        cycle_gamma_range: Cycle基因的gamma范围
        cycle_angular_sigma: Cycle基因的angular_sigma（高斯峰角宽度，单位：弧度）
                            - 0.3 ~ 0.5: 锐利（约17~28度），适合热图
                            - 0.8 ~ 1.0: 宽阔（约45~60度），适合PCA/UMAP建图
                            - 默认值0.8
        edge_beta_range: Edge基因的beta范围
        edge_gamma_range: Edge基因的gamma范围
        sigmoid_center_range: Sigmoid center范围
        sigmoid_slope_range: Sigmoid slope范围
        impulse_center_range: Impulse center范围
        impulse_width_range: Impulse width范围
        edge_gene_distribution: {edge_idx: {'sigmoid': int, 'sigmoid_reverse': int, 'sticky': int, 'impulse': int}}
                               为每条边精确指定基因分布
                               - 'sigmoid': 正向sigmoid基因（sticky=False）
                               - 'sigmoid_reverse': 反向sigmoid基因（sticky=False）
                               - 'sticky': 正向sigmoid基因（sticky=True，独立类型）
                               - 'impulse': impulse基因
                               例如: {0: {'sigmoid': 30, 'sigmoid_reverse': 10, 'sticky': 20, 'impulse': 20}}
                               如果提供了此参数，将忽略下面的自动分配策略参数
        first_edge_has_background: Edge 0 是否包含背景噪音基因（仅在 edge_gene_distribution 为 None 时使用）
        later_edges_sticky: 后续Edge是否使用sticky模式（仅在 edge_gene_distribution 为 None 时使用）
        default_edge_pattern: 默认Edge模式（'sigmoid' 或 'impulse'），用于所有未明确指定的Edge
                             （仅在 edge_gene_distribution 为 None 时使用）
        edge_pattern: {edge_idx: 'sigmoid'|'impulse'} 为每个Edge指定模式（覆盖默认值）
                     - 如果某个Edge未在此字典中，使用 default_edge_pattern
                     - 注意：如果 first_edge_has_background=True，Edge 0 会同时包含 sigmoid 和 impulse
                     - （仅在 edge_gene_distribution 为 None 时使用）
        n_noise_genes: 噪声基因数量（noise类型，白噪声，不依赖于edge）
        noise_alpha_range: 噪声基因的alpha范围（转录速率，非常小的值），默认(0.01, 0.05)
        noise_beta_range: 噪声基因的beta范围（如果None，使用edge_beta_range）
        noise_gamma_range: 噪声基因的gamma范围（如果None，使用edge_gamma_range）
        default_include_cycle_rim: 默认是否将圆环视为Edge的一部分
        edge_include_cycle_rim: {edge_idx: bool} 每个Edge的include_cycle_rim设置（覆盖默认值）
        
    Returns:
        完整的配置字典 {'genes': [...]}
        
    Raises:
        ImportError: 如果无法导入 UniversalGraphSystem
    """
    if not HAS_SYSTEM_MODULE:
        raise ImportError("无法导入 UniversalGraphSystem，请确保系统模块可用")
    
    generator = GeneConfigGenerator(seed=seed)
    
    # 1. 扫描 System 中的 Cycle Node
    cycle_genes_config = {}
    for node in system.nodes:
        # 检查是否为 CycleNode（通过类名或类型判断）
        is_cycle = (
            isinstance(node, CycleNode) if HAS_SYSTEM_MODULE else
            "Cycle" in node.__class__.__name__ or 
            (hasattr(node, 'id') and "Cycle" in node.id)
        )
        
        if is_cycle:
            node_id = node.id if hasattr(node, 'id') else str(node)
            cycle_genes_config[node_id] = n_genes_per_cycle

    # 2. 扫描 System 中的 Edge
    edge_genes_config = {}
    
    if edge_gene_distribution is not None:
        # 使用精确的基因分布配置
        for i, edge in enumerate(system.edges):
            # 确定该Edge的include_cycle_rim设置
            if edge_include_cycle_rim is not None and i in edge_include_cycle_rim:
                include_rim = edge_include_cycle_rim[i]
            else:
                include_rim = default_include_cycle_rim
            
            # 获取该Edge的基因分布
            if i in edge_gene_distribution:
                dist = edge_gene_distribution[i]
                configs = []
                
                # 添加 sigmoid 基因（正向，sticky=False）
                if dist.get('sigmoid', 0) > 0:
                    configs.append({
                        'n_genes': dist['sigmoid'],
                        'pattern': 'sigmoid',
                        'reverse': False,
                        'sticky': False,
                        'center_range': sigmoid_center_range,
                        'slope_range': sigmoid_slope_range,
                        'include_cycle_rim': include_rim
                    })
                
                # 添加 sigmoid_reverse 基因（反向，sticky=False）
                if dist.get('sigmoid_reverse', 0) > 0:
                    configs.append({
                        'n_genes': dist['sigmoid_reverse'],
                        'pattern': 'sigmoid',
                        'reverse': True,
                        'sticky': False,  # reverse 通常不使用 sticky
                        'center_range': sigmoid_center_range,
                        'slope_range': sigmoid_slope_range,
                        'include_cycle_rim': include_rim
                    })
                
                # 添加 sticky 基因（正向sigmoid，sticky=True，独立类型）
                if dist.get('sticky', 0) > 0:
                    configs.append({
                        'n_genes': dist['sticky'],
                        'pattern': 'sigmoid',
                        'reverse': False,
                        'sticky': True,  # sticky模式
                        'center_range': sigmoid_center_range,
                        'slope_range': sigmoid_slope_range,
                        'include_cycle_rim': include_rim
                    })
                
                # 添加 impulse 基因
                if dist.get('impulse', 0) > 0:
                    configs.append({
                        'n_genes': dist['impulse'],
                        'pattern': 'impulse',
                        'sticky': False,
                        'center_range': impulse_center_range,
                        'width_range': impulse_width_range,
                        'include_cycle_rim': include_rim
                    })
                
                # 如果只有一个配置，直接使用字典；否则使用列表
                if len(configs) == 1:
                    edge_genes_config[i] = configs[0]
                elif len(configs) > 1:
                    edge_genes_config[i] = configs
                # 如果 dist 中所有值都是0，则跳过该Edge（不添加任何基因）
            else:
                # 如果该Edge未在 edge_gene_distribution 中指定，跳过（不添加任何基因）
                pass
    else:
        # 使用原来的自动分配策略
        for i, edge in enumerate(system.edges):
            # 确定该Edge的include_cycle_rim设置
            if edge_include_cycle_rim is not None and i in edge_include_cycle_rim:
                include_rim = edge_include_cycle_rim[i]
            else:
                include_rim = default_include_cycle_rim
            
            # 确定该Edge的模式
            if edge_pattern is not None and i in edge_pattern:
                pattern = edge_pattern[i]
                if pattern not in ['sigmoid', 'impulse']:
                    raise ValueError(f"Edge {i} 的模式必须是 'sigmoid' 或 'impulse'，得到 '{pattern}'")
            else:
                pattern = default_edge_pattern
            
            # 自动分配策略
            if i == 0 and first_edge_has_background:
                # Edge 0: 包含 sigmoid（启动）和 impulse（背景噪音）
                # 注意：impulse 不支持 reverse，如果需要反向效果，使用 sigmoid 模式
                edge_genes_config[i] = [
                    {
                        'n_genes': n_genes_per_edge,
                        'pattern': 'sigmoid',
                        'reverse': False,
                        'sticky': False,
                        'center_range': sigmoid_center_range,
                        'slope_range': sigmoid_slope_range,
                        'include_cycle_rim': include_rim
                    },
                    {
                        'n_genes': n_genes_per_edge,
                        'pattern': 'impulse',
                        # impulse 不支持 reverse，如果需要反向效果，使用 sigmoid 模式
                        'sticky': False,
                        'center_range': impulse_center_range,
                        'width_range': impulse_width_range,
                        'include_cycle_rim': include_rim
                    }
                ]
            else:
                # 其他 Edge: 使用指定的模式（或默认模式）
                if pattern == 'sigmoid':
                    edge_genes_config[i] = {
                        'n_genes': n_genes_per_edge,
                        'pattern': 'sigmoid',
                        'reverse': False,
                        'sticky': later_edges_sticky,  # 分化后基因保持开启
                        'center_range': sigmoid_center_range,
                        'slope_range': sigmoid_slope_range,
                        'include_cycle_rim': include_rim
                    }
                else:  # impulse
                    edge_genes_config[i] = {
                        'n_genes': n_genes_per_edge,
                        'pattern': 'impulse',
                        'sticky': False,  # impulse 通常不使用 sticky
                        'center_range': impulse_center_range,
                        'width_range': impulse_width_range,
                        'include_cycle_rim': include_rim
                    }

    # 3. 生成完整配置（噪声基因现在作为noise类型，不依赖于edge）
    full_config = generator.generate_config(
        cycle_genes=cycle_genes_config,
        edge_genes=edge_genes_config,
        cycle_beta_range=cycle_beta_range,
        cycle_gamma_range=cycle_gamma_range,
        cycle_angular_sigma=cycle_angular_sigma,
        edge_beta_range=edge_beta_range,
        edge_gamma_range=edge_gamma_range,
        noise_genes=n_noise_genes if n_noise_genes > 0 else None,
        noise_alpha_range=noise_alpha_range,
        noise_beta_range=noise_beta_range if noise_beta_range is not None else edge_beta_range,
        noise_gamma_range=noise_gamma_range if noise_gamma_range is not None else edge_gamma_range,
        default_sigmoid_center_range=sigmoid_center_range,
        default_sigmoid_slope_range=sigmoid_slope_range,
        default_impulse_center_range=impulse_center_range,
        default_impulse_width_range=impulse_width_range
    )
    
    return full_config
