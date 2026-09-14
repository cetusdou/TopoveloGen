"""
基因配置加载器
从YAML配置文件批量添加基因到GeneSynthesizer
"""

import yaml
import numpy as np
from typing import List, Dict, Any, Optional
from .synthesizer_gene import GeneSynthesizer


def load_gene_config(config_path: str) -> List[Dict[str, Any]]:
    """
    从YAML文件加载基因配置
    
    Args:
        config_path: YAML配置文件路径
        
    Returns:
        基因配置列表
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config.get('genes', [])


def add_genes_from_config(
    synthesizer: GeneSynthesizer,
    config_path: str,
    verbose: bool = True
) -> Dict[str, int]:
    """
    从配置文件批量添加基因到GeneSynthesizer
    
    Args:
        synthesizer: GeneSynthesizer实例
        config_path: YAML配置文件路径
        verbose: 是否打印详细信息
        
    Returns:
        基因名称到索引的映射字典
    """
    genes_config = load_gene_config(config_path)
    
    gene_name_to_idx = {}
    cycle_phases = {}  # 用于批量添加cycle基因
    noise_genes_list = []  # 用于批量添加noise基因
    
    # 第一遍：收集cycle基因的相位和noise基因
    for gene_config in genes_config:
        if gene_config.get('type') == 'cycle':
            node_id = gene_config.get('node_id')
            phase = gene_config.get('phase', 0.0)
            name = gene_config.get('name', f'cycle_{len(cycle_phases)}')
            
            if node_id not in cycle_phases:
                cycle_phases[node_id] = {'phases': [], 'names': []}
            
            cycle_phases[node_id]['phases'].append(phase)
            cycle_phases[node_id]['names'].append(name)
        elif gene_config.get('type') == 'noise':
            noise_genes_list.append(gene_config)
    
    # 批量添加cycle基因
    for node_id, data in cycle_phases.items():
        phases = data['phases']
        names = data['names']
        
        # 添加cycle基因
        start_idx = synthesizer.current_gene_idx
        synthesizer.add_cycle_gene(node_id=node_id, n_genes=len(phases), phase=phases)
        
        # 记录基因名称到索引的映射
        for i, name in enumerate(names):
            gene_name_to_idx[name] = start_idx + i
        
        if verbose:
            print(f"  ✓ Added {len(phases)} cycle genes to node '{node_id}': {names}")
    
    # 批量添加noise基因
    if noise_genes_list:
        # 收集所有noise基因的alpha值
        alpha_values = []
        names = []
        for gene_config in noise_genes_list:
            alpha = gene_config.get('alpha', 0.03)
            alpha_values.append(alpha)
            names.append(gene_config.get('name', f'noise_{len(names)}'))
        
        # 计算alpha范围（基于所有alpha值的范围）
        if alpha_values:
            alpha_min = min(alpha_values) * 0.8
            alpha_max = max(alpha_values) * 1.2
            alpha_range = (alpha_min, alpha_max)
        else:
            alpha_range = (0.01, 0.05)
        
        start_idx = synthesizer.current_gene_idx
        synthesizer.add_noise_gene(
            n_genes=len(noise_genes_list),
            alpha_range=alpha_range
        )
        
        # 记录基因名称到索引的映射
        for i, name in enumerate(names):
            gene_name_to_idx[name] = start_idx + i
        
        if verbose:
            print(f"  ✓ Added {len(noise_genes_list)} noise genes: {names[:5]}{'...' if len(names) > 5 else ''}")
    
    # 第二遍：添加其他类型的基因
    for gene_config in genes_config:
        gene_type = gene_config.get('type')
        name = gene_config.get('name', f'gene_{synthesizer.current_gene_idx}')
        
        if gene_type == 'cycle' or gene_type == 'noise':
            continue  # 已经在第一遍处理了
        
        if gene_type in ['sigmoid', 'impulse']:
            edge_idx = gene_config.get('edge_idx')
            if edge_idx is None:
                raise ValueError(f"Gene '{name}': edge_idx is required for {gene_type} genes")
            
            sticky = gene_config.get('sticky', False)
            reverse = gene_config.get('reverse', False)
            
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
            start_idx = synthesizer.current_gene_idx
            synthesizer.add_trend_gene(
                edge_idx=edge_idx,
                n_genes=1,
                pattern=gene_type,
                reverse=reverse,
                sticky=sticky,
                params=params if params else None
            )
            
            gene_name_to_idx[name] = start_idx
            
            if verbose:
                gene_type_str = f"{gene_type}{' (sticky)' if sticky else ''}"
                print(f"  ✓ Added {gene_type_str} gene '{name}' to edge {edge_idx}")
        else:
            raise ValueError(f"Unknown gene type: {gene_type}")
    
    if verbose:
        print(f"\n✓ Total genes added: {len(gene_name_to_idx)}")
        print(f"  Gene name to index mapping:")
        for name, idx in sorted(gene_name_to_idx.items(), key=lambda x: x[1]):
            print(f"    {name}: Gene {idx}")
    
    return gene_name_to_idx


def get_gene_info_from_config(config_path: str) -> Dict[str, Any]:
    """
    从配置文件获取基因信息摘要
    
    Args:
        config_path: YAML配置文件路径
        
    Returns:
        包含基因统计信息的字典
    """
    genes_config = load_gene_config(config_path)
    
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

