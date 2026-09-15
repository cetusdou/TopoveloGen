
# ===== Original notebook cell 1 =====

# 导入必要的库
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Arrow
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from pathlib import Path
import sys
from umap import UMAP

# 添加项目根目录到路径
project_root = Path.cwd().parent if Path.cwd().name == 'figure_notebook' else Path.cwd()
sys.path.insert(0, str(project_root))
print(f"项目根目录: {project_root}")

from experiments.figure_style import apply_publication_style

apply_publication_style()
plt.rcParams.update(
    {
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.compression": 9,
    }
)

OUT_DIR = project_root / 'results' / 'figure3_series'
FIG_DIR = OUT_DIR / 'figures'
DATA_DIR = OUT_DIR / 'data'
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
print(f"图片输出目录: {FIG_DIR}")

COLOR_DICT = {
    'deep_blue_gray': '#396185',
    'blue_gray': '#6981a3',
    'light_blue': '#93b1d1',
    'lighter_blue': '#a8c9e8',
    'very_light_blue': '#bde0fe',
    'light_purple': '#ded4ee',
    'light_pink': '#ffc8dd',
    'pink': '#f1b5c8',
    'deep_pink': '#e2a2b3',
    'deeper_pink': '#db8c9f',
    'deepest_pink': '#cb7892',
}
COLOR_LIST = [
    COLOR_DICT['deep_blue_gray'],
    COLOR_DICT['blue_gray'],
    COLOR_DICT['light_blue'],
    COLOR_DICT['light_pink'],
    COLOR_DICT['deep_pink'],
    COLOR_DICT['deepest_pink'],
]
if 'TopoVelo_grad' not in plt.colormaps():
    plt.colormaps.register(
        LinearSegmentedColormap.from_list(
            'TopoVelo_grad',
            [COLOR_DICT['deep_blue_gray'], COLOR_DICT['light_blue'], COLOR_DICT['light_pink'], COLOR_DICT['deepest_pink']],
        )
    )
if 'TopoVelo_cat' not in plt.colormaps():
    plt.colormaps.register(ListedColormap(COLOR_LIST, name='TopoVelo_cat'))

_FIG_COUNTER = 0
def save_current_figure(stem=None):
    global _FIG_COUNTER
    _FIG_COUNTER += 1
    if stem is None:
        stem = f'Figure3Series_{_FIG_COUNTER:02d}'
    fig = plt.gcf()
    if not fig.axes:
        plt.close(fig)
        return
    fig.savefig(FIG_DIR / f'{stem}.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / f'{stem}.svg', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {stem}.png/.pdf/.svg")

# 导入数据生成模块
try:
    from data_generation_module import GraphNode, CycleNode, GraphEdge, UniversalGraphSystem, SkeletonNode
    from data_generation_module import GraphSimulator
    from data_generation_module import GeneSynthesizer
    HAS_DATA_MODULE = True
except ImportError as e:
    print(f"警告: 无法导入数据生成模块: {e}")
    HAS_DATA_MODULE = False

print("✓ 库导入完成")
print("✓ Publication style 已加载（styles/nature_science.mplstyle）")

# ===== Original notebook cell 2 =====

# 生成Cycle-Linear拓扑数据
def generate_cycle_linear_data(N_samples=2000, n_steps=100, dt=0.05, noise_level=0.15, seed=42):
    """生成Cycle-Linear拓扑的模拟数据"""
    np.random.seed(seed)
    np.random.seed(seed)
    
    # 创建Cycle-Linear拓扑
    nodes = [
        GraphNode("Start", [-3.0, 0], potential=-1.0, sigma=0.8),
        CycleNode("Cycle", [0.0, 0], potential=-3.0, sigma=0.5, 
                    radius=1.0, omega=4.0, clockwise=False),
        GraphNode("Exit", [3.0, 0], potential=-4.0, sigma=0.5)
    ]
    edges = [
        GraphEdge(nodes[0], nodes[1]),  # Start -> Cycle
        GraphEdge(nodes[1], nodes[2])   # Cycle -> Exit
    ]
    start_node = "Start"
    
    system = UniversalGraphSystem(nodes, edges, fusion_temp=0.05)
    traj = GraphSimulator.run(system, start_node, n_cells=N_samples, 
                                steps=n_steps, dt=dt, noise_level=noise_level)
    
    if isinstance(traj, np.ndarray):
        z_trajectory = traj
    else:
        z_trajectory = np.array(traj)
    
    z_data = z_trajectory[-1]  # 最后一个时间点的坐标
    time_points = np.arange(n_steps + 1) * dt
    
    # 计算pseudotime和branch
    dist_to_origin = np.linalg.norm(z_data, axis=1)
    on_cycle = dist_to_origin < 1.3
    pseudotime = np.zeros(N_samples)
    pseudotime[on_cycle] = np.arctan2(z_data[on_cycle, 1], z_data[on_cycle, 0])
    pseudotime[on_cycle] = (pseudotime[on_cycle] + np.pi) / (2 * np.pi)
    pseudotime[~on_cycle] = 1.0 + (z_data[~on_cycle, 0] - 1.0) / 1.5
    branch = (~on_cycle).astype(int)
    
    return {
        'z_data': z_data,
        'z_trajectory': z_trajectory,
        'pseudotime': pseudotime,
        'branch': branch,
        'time_points': time_points,
        'system': system
    }

# 生成数据
print("生成Cycle-Linear拓扑数据...")
topology_data = generate_cycle_linear_data(N_samples=500, n_steps=150, dt=0.05, seed=42)
print(f"✓ 数据生成完成: {topology_data['z_data'].shape[0]} 个细胞")

# ===== Original notebook cell 3 =====

plt.scatter(topology_data['time_points'].repeat(topology_data['z_data'].shape[0],axis=0),topology_data['z_trajectory'][:,:,0],s=10,alpha=0.01)

# ===== Original notebook cell 4 =====

# 自动生成基因配置文件并生成基因数据
import yaml
from data_generation_module.generate_gene_config import auto_generate_config_from_system

# ---------------------------------------------------------
# 第一步：获取系统对象
# ---------------------------------------------------------
system = topology_data['system']
print(f"当前系统包含: {len(system.nodes)} 个节点, {len(system.edges)} 条边")

# 检查Cycle节点
cycle_nodes = [n for n in system.nodes if isinstance(n, CycleNode)]
print(f"  其中Cycle节点: {len(cycle_nodes)} 个")

# ---------------------------------------------------------
# 第二步：检查配置文件是否存在，如果存在则直接加载，否则生成
# ---------------------------------------------------------
config_path = DATA_DIR / "auto_topology_config.yaml"

if Path(config_path).exists():
    print(f"\n检测到配置文件已存在: {config_path}")
    print("正在加载现有配置...")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    total_genes = len(config_dict['genes'])
    print(f"✓ 配置加载完成，共 {total_genes} 个基因")
else:
    print("\n配置文件不存在，正在根据拓扑结构自动生成基因配置...")
    
    # 调用自动生成函数（使用方式1：精确控制每条边的基因分布）
    config_dict = auto_generate_config_from_system(
        system,
        n_genes_per_cycle=30,      # 每个Cycle节点分配 300 个基因
        seed=42,                    # 固定随机种子
        
        # 动力学参数范围
        cycle_beta_range=(3.6, 5.0),
        cycle_gamma_range=(3.0, 4.0),
        edge_beta_range=(1.6, 3.0),  # 让分化基因的爆发速度有差异
        edge_gamma_range=(0.8, 2.0),
        cycle_angular_sigma = 1.0,
        # Sigmoid/Impulse 参数范围
        sigmoid_center_range=(0.2, 0.8),
        sigmoid_slope_range=(5.0, 15.0),
        impulse_center_range=(0.1, 0.9),
        impulse_width_range=(0.05, 0.15),
        
        # 精确控制每条边的基因分布（方式1）
        # sticky作为独立类型，类似sigmoid但sticky=True
        edge_gene_distribution={
            0: {  # Edge 0: Start -> Cycle
                'sigmoid': 5,          # 40个正向sigmoid基因（启动基因，sticky=False）
                'sigmoid_reverse': 2,   # 20个反向sigmoid基因
                'sticky': 3,            # 10个sticky基因（正向sigmoid，sticky=True）
                'impulse': 3           # 30个impulse基因（背景噪音）
            },
            1: {  # Edge 1: Cycle -> Exit
                'sigmoid': 5,          # 30个正向sigmoid基因（分化基因，sticky=False）
                'sigmoid_reverse': 2,   # 0个反向sigmoid基因
                'sticky': 0,           # 20个sticky基因（分化后保持开启）
                'impulse': 5             # 0个impulse基因
            }
        },
        
        # 噪声基因配置（纯噪声基因，noise类型，白噪声，不依赖于edge）
        n_noise_genes=5,                 # 噪声基因数量（noise类型）
        noise_alpha_range=(0.01, 0.03),   # 噪声基因的alpha范围（转录速率，非常小的值）
        
        # Cycle Rim 设置
        # include_cycle_rim控制：如果Edge连接到CycleNode，是否将圆环(Rim)视为Edge的一部分
        # - True: 将圆环视为Edge的一部分（细胞在圆环上时，表达依然活跃）
        # - False: 不将圆环视为Edge的一部分（只计算到线段的距离，圆环上的细胞表达会降低）
        default_include_cycle_rim=True,  # 默认不将圆环视为Edge的一部分，避免圆环上一直高表达
        edge_include_cycle_rim={
            # 0: False,  # Edge 0 (Start -> Cycle): 不将圆环视为Edge的一部分
            # 1: False,  # Edge 1 (Cycle -> Exit): 不将圆环视为Edge的一部分
            # 如果需要某个Edge将圆环视为Edge的一部分，可以单独设置：
            # 0: True,   # Edge 0: 将圆环视为Edge的一部分
        },
    )
    
    # 计算总基因数
    total_genes = len(config_dict['genes'])
    print(f"✓ 配置生成完成，共 {total_genes} 个基因")
    
    # 保存为 YAML 文件
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"✓ 配置文件已保存至: {config_path}")

# 统计基因类型（包括sigmoid_reverse、sticky和noise）
stats = {'cycle': 0, 'sigmoid': 0, 'impulse': 0, 'noise': 0}
stats_by_edge = {}  # 统计每个Edge的基因数
for gene in config_dict['genes']:
    gene_type = gene['type']
    # 如果类型不在stats中，先添加
    if gene_type not in stats:
        stats[gene_type] = 0
    stats[gene_type] += 1
    
    if gene_type in ['sigmoid', 'impulse']:
        edge_idx = gene.get('edge_idx')
        if edge_idx is not None:
            if edge_idx not in stats_by_edge:
                stats_by_edge[edge_idx] = {'sigmoid': 0, 'sigmoid_reverse': 0, 'sticky': 0, 'impulse': 0}
            
            # 区分正向sigmoid、反向sigmoid和sticky
            if gene_type == 'sigmoid':
                if gene.get('sticky', False):
                    stats_by_edge[edge_idx]['sticky'] += 1
                elif gene.get('reverse', False):
                    stats_by_edge[edge_idx]['sigmoid_reverse'] += 1
                else:
                    stats_by_edge[edge_idx]['sigmoid'] += 1
            else:  # impulse
                stats_by_edge[edge_idx]['impulse'] += 1

print(f"  基因类型分布: {stats}")
print(f"  各Edge基因分布:")
for edge_idx, edge_stats in sorted(stats_by_edge.items()):
    total = sum(edge_stats())
    print(f"    Edge {edge_idx}: {edge_stats} (总计: {total})")

# 统计 include_cycle_rim 设置
rim_stats = {'include': 0, 'exclude': 0}
for gene in config_dict['genes']:
    if gene['type'] in ['sigmoid', 'impulse']:
        include_rim = gene.get('include_cycle_rim', True)
        if include_rim:
            rim_stats['include'] += 1
        else:
            rim_stats['exclude'] += 1
print(f"  Cycle Rim设置: {rim_stats}")

# ---------------------------------------------------------
# 第三步：初始化合成器并加载配置
# ---------------------------------------------------------
print("\n初始化 GeneSynthesizer 并加载配置...")
synth = GeneSynthesizer(
    system=system,
    n_genes_total=total_genes,  # 使用配置中的总数
    seed=42,
    dropout_prob=0.0,  # Dropout概率（15%的基础dropout率）
    dropout_slope=0.0   # Dropout斜率（低表达基因更容易dropout）
)

# 加载配置
synth.load_from_config(config_path, verbose=True)

# ---------------------------------------------------------
# 第四步：生成基因表达数据
# ---------------------------------------------------------
print("\n生成基因表达数据...")





# 生成动力学数据（unspliced和spliced）
z_trajectory = topology_data['z_trajectory']  # [n_steps+1, N_samples, 2]
z_trajectory_tensor = np.asarray(z_trajectory, dtype=np.float32)
dt = topology_data['time_points'][1] - topology_data['time_points'][0]

print(f"\n轨迹信息:")
print(f"  z_trajectory shape: {z_trajectory.shape}")
print(f"  dt: {dt:.4f}")
print(f"  n_steps: {z_trajectory.shape[0] - 1}")

if hasattr(synth, 'generate_kinetics'):
    print("\n生成动力学数据（unspliced/spliced）...")
    kinetics_result = synth.generate_kinetics(
        z_trajectory_tensor, 
        dt=dt, 
        simulate_counts=False,  # 生成带dropout噪声的计数数据
        scale=1.5  # 空间门控的缩放因子
    )
    
    n_cells = topology_data['z_data'].shape[0]
    
    # 获取最后一个时间点的数据（最后n_cells行）
    u_all = kinetics_result['unspliced'][-n_cells:]  # [n_cells, n_genes]
    s_all = kinetics_result['spliced'][-n_cells:]   # [n_cells, n_genes]
    
    print(f"✓ 动力学数据生成完成（带dropout噪声的计数数据）")
    print(f"  u_all shape: {u_all.shape}")
    print(f"  s_all shape: {s_all.shape}")
    print(f"  u range: [{u_all.min():.2f}, {u_all.max():.2f}], 稀疏度: {(u_all == 0).mean():.2%}")
    print(f"  s range: [{s_all.min():.2f}, {s_all.max():.2f}], 稀疏度: {(s_all == 0).mean():.2%}")
    
    # 检查是否有 velocity 和 alpha 数据
    if 'velocity' in kinetics_result:
        vel_all = kinetics_result['velocity'][-n_cells:] if isinstance(kinetics_result['velocity'], np.ndarray) else kinetics_result['velocity'][-n_cells:]
        print(f"  velocity range: [{vel_all.min():.2f}, {vel_all.max():.2f}]")
    
    if 'alpha' in kinetics_result:
        alpha_all = kinetics_result['alpha'][-n_cells:] if isinstance(kinetics_result['alpha'], np.ndarray) else kinetics_result['alpha'][-n_cells:]
        print(f"  alpha (transcription rate) range: [{alpha_all.min():.2f}, {alpha_all.max():.2f}]")
    
    # ---------------------------------------------------------
    # 保存 kinetics_result
    # ---------------------------------------------------------
    print("\n保存 kinetics_result...")
    kinetics_save_path = "kinetics_result.npz"
    
    # 将 np.ndarray 转换为 numpy 数组
    save_dict = {}
    for key in ['unspliced', 'spliced', 'velocity', 'alpha']:
        if key in kinetics_result:
            if isinstance(kinetics_result[key], np.ndarray):
                save_dict[key] = kinetics_result[key]
            else:
                save_dict[key] = kinetics_result[key]
    
    # 添加元数据
    save_dict['n_cells'] = n_cells
    save_dict['n_genes'] = total_genes
    save_dict['n_steps'] = z_trajectory.shape[0] - 1
    save_dict['dt'] = dt
    
    # # 保存为 .npz 文件（numpy 压缩格式）
    # np.savez_compressed(kinetics_save_path, **save_dict)
    
    # print(f"✓ kinetics_result 已保存至: {kinetics_save_path}")
    # print(f"  包含数据: {list(save_dict.keys())}")
    # for key in ['unspliced', 'spliced', 'velocity', 'alpha']:
    #     if key in save_dict:
    #         print(f"    {key} shape: {save_dict[key].shape}")
    
else:
    print("警告: GeneSynthesizer 不支持 generate_kinetics 方法")
    u_all = None
    s_all = None
    kinetics_result = None

print("\n✓ 所有数据生成完成！")
print(f"\n数据摘要:")
print(f"  - 总基因数: {total_genes}")
print(f"  - 细胞数: {n_cells}")

print(f"  - Unspliced数据: {u_all.shape if u_all is not None else 'N/A'}")
print(f"  - Spliced数据: {s_all.shape if s_all is not None else 'N/A'}")

# ===== Original notebook cell 5 =====

z_trajectory_sample_indices = np.random.randint(0,z_trajectory.shape[0]*z_trajectory.shape[1],size=5000)
fig,ax = plt.subplots(1,3,figsize=(18,3))  # 增加宽度以容纳colorbar
ax = ax.flatten()
gene_idx =16

# 第一个子图：alpha
color1 = kinetics_result['alpha'][z_trajectory_sample_indices,gene_idx]
scatter1 = ax[0].scatter(z_trajectory.reshape(-1,2)[z_trajectory_sample_indices,0],
                        z_trajectory.reshape(-1,2)[z_trajectory_sample_indices,1],
                        s=10,alpha=0.5,c=color1,cmap='TopoVelo_grad')
ax[0].set_title('alpha of cycle gene 5')
ax[0].set_aspect('equal')
ax[0].axis('off')
plt.colorbar(scatter1, ax=ax[0], label='Alpha (transcription rate)')

# 第二个子图：unspliced
color2 = kinetics_result['unspliced'][z_trajectory_sample_indices,gene_idx]
scatter2 = ax[1].scatter(z_trajectory.reshape(-1,2)[z_trajectory_sample_indices,0],
                        z_trajectory.reshape(-1,2)[z_trajectory_sample_indices,1],
                        s=10,alpha=0.5,c=color2,cmap='TopoVelo_grad')
ax[1].set_title('unspliced RNA of cycle gene 5')
ax[1].set_aspect('equal')
ax[1].axis('off')
plt.colorbar(scatter2, ax=ax[1], label='Unspliced RNA')

# 第三个子图：spliced
color3 = kinetics_result['spliced'][z_trajectory_sample_indices,gene_idx]
scatter3 = ax[2].scatter(z_trajectory.reshape(-1,2)[z_trajectory_sample_indices,0],
                        z_trajectory.reshape(-1,2)[z_trajectory_sample_indices,1],
                        s=10,alpha=0.5,c=color3,cmap='TopoVelo_grad')
ax[2].set_title('spliced RNA of cycle gene 5')
ax[2].set_aspect('equal')
ax[2].axis('off')
plt.colorbar(scatter3, ax=ax[2], label='Spliced RNA')

plt.tight_layout()
save_current_figure('Figure3Series_gene_fields')

# ===== Original notebook cell 6 =====

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP

print("绘制Panel A: Manifold Reconstruction (PCA + UMAP)...")

# ==========================================
# 1. 数据提取 (Extract)
# ==========================================

# A. 提取三个矩阵
X_s = kinetics_result['spliced']
X_u = kinetics_result['unspliced']
X_a = kinetics_result['alpha'] # 提取 Alpha

# 转换为 numpy
if hasattr(X_s, 'numpy'): X_s = X_s
if hasattr(X_u, 'numpy'): X_u = X_u
if hasattr(X_a, 'numpy'): X_a = X_a

# 确保形状一致 (N_samples, n_genes)
X_s = X_s.reshape(-1, X_s.shape[1]) # 假设 480 是基因数，或者使用 X_s.shape[1]
X_u = X_u.reshape(-1, X_u.shape[1])
X_a = X_a.reshape(-1, X_a.shape[1])

# B. 获取几何坐标
z_traj = topology_data['z_trajectory']
z_flat = z_traj.reshape(-1, 2)

# C. 对齐行数 (以防万一)
min_len = min(X_s.shape[0], z_flat.shape[0])
X_s = X_s[:min_len]
X_u = X_u[:min_len]
X_a = X_a[:min_len]
z_flat = z_flat[:min_len]

# ==========================================
# 2. 随机采样 (Sample Indices)
# ==========================================
# 先生成索引，再提取数据，这样内存更省
total_points = min_len
sample_size = 10000

if total_points > sample_size:
    indices = np.random.choice(total_points, sample_size, replace=False)
    # 使用相同索引采样所有模态
    X_s_sub = X_s[indices]
    X_u_sub = X_u[indices]
    X_a_sub = X_a[indices]
    z_sub = z_flat[indices]
else:
    X_s_sub, X_u_sub, X_a_sub, z_sub = X_s, X_u, X_a, z_flat

print(f"采样完成: {X_s_sub.shape}")

# ==========================================
# 3. 独立预处理 (Independent Preprocessing)
# ==========================================
# 这一步是关键：分别处理，消除不同物理量的量纲差异

def preprocess_modality(X_data, name):
    """辅助函数：对单一模态进行 Log1p 和 StandardScaler"""
    # 1. Log1p: 处理偏态分布，压制极大值
    X_log = np.log1p(X_data)
    
    # 2. Scale: 转换为 Z-score (均值0，方差1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)
    print(f" -> {name} Standardized. Mean~{X_scaled.mean():.2f}, Std~{X_scaled.std():.2f}")
    return X_scaled

print("开始独立标准化...")
X_s_scaled = preprocess_modality(X_s_sub, "Spliced")
X_u_scaled = preprocess_modality(X_u_sub, "Unspliced")
X_a_scaled = preprocess_modality(X_a_sub, "Alpha")

# ==========================================
# 4. 特征融合 (Fusion)
# ==========================================

# ==========================================
# 5. 定义颜色标签
# ==========================================
dist_to_origin = np.linalg.norm(z_sub, axis=1)
# 稍微放宽范围防止断裂
on_cycle = (dist_to_origin < 1.35) & (dist_to_origin > 0.65) 
on_edge0 = (z_sub[:, 0] < -1.0) & (~on_cycle)
on_edge1 = (~on_cycle) & (~on_edge0)

point_labels = np.zeros(len(z_sub), dtype=int)
point_labels[on_edge0] = 0
point_labels[on_cycle] = 1
point_labels[on_edge1] = 2

colors_map = ['#2E86AB', '#A23B72', '#F18F01']
label_names = ['Edge0 (Start)', 'Cycle', 'Edge1 (End)']

# ==========================================
# 6. 降维与绘图
# ==========================================

# --- PCA ---
print("计算 PCA...")
# 使用融合后的数据
pca_spliced = PCA(n_components=4, random_state=42)
z_pca_spliced = pca_spliced.fit_transform(X_s_scaled) 
var_ratio_spliced = pca_spliced.explained_variance_ratio_


pca_unspliced = PCA(n_components=4, random_state=42)
z_pca_unspliced = pca_unspliced.fit_transform(X_u_scaled) 
var_ratio_unspliced = pca_unspliced.explained_variance_ratio_

pca_alpha = PCA(n_components=4, random_state=42)
z_pca_alpha = pca_alpha.fit_transform(X_a_scaled) 
var_ratio_alpha = pca_alpha.explained_variance_ratio_


# --- UMAP ---
# (取消注释以运行 UMAP)
print("计算 UMAP...")
# umap_model = UMAP(
#    n_components=2, 
#    n_neighbors=50,  # 这里的 neighbor 建议设大一点，因为增加了 alpha 维度，稀疏性变了
#    min_dist=0.5, 
#    metric='euclidean',
#    random_state=42
# )
# z_umap = umap_model.fit_transform(X_final)

# --- 绘图 ---
fig, axes = plt.subplots(3, 3, figsize=(16, 16))

def plot_scatter(ax, coords, title, xlabel, ylabel):
    for i, color in enumerate(colors_map):
        mask = point_labels == i
        if np.sum(mask) > 0:
            sub_coords = coords[mask]
            ax.scatter(sub_coords[:, 0], sub_coords[:, 1], 
                      c=color, label=label_names[i], 
                      s=10, alpha=0.6, edgecolors='none')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=10, markerscale=2)
    ax.grid(True, alpha=0.2, linestyle='--')

for i in range(3):
    for j in range(3):
        if i==0:
            
            plot_scatter(axes[i,j], z_pca_spliced[:,[j,j+1]], 
             f'A. PCA Visualization\n(Spliced PC{j+1}+PC{j+2})', 
             f'PC{j+1} ({var_ratio_spliced[j]:.1%} variance)', 
             f'PC{j+2} ({var_ratio_spliced[j+1]:.1%} variance)')
        elif i==1:
            plot_scatter(axes[i,j], z_pca_unspliced[:,[j,j+1]], 
             f'A. PCA Visualization\n(Unspliced PC{j+1}+PC{j+2})', 
             f'PC{j+1} ({var_ratio_unspliced[j]:.1%} variance)', 
             f'PC{j+2} ({var_ratio_unspliced[j+1]:.1%} variance)')
        elif i==2:
            plot_scatter(axes[i,j], z_pca_alpha[:,[j,j+1]], 
             f'A. PCA Visualization\n(Alpha PC{j+1}+PC{j+2})', 
             f'PC{j+1} ({var_ratio_alpha[j]:.1%} variance)', 
             f'PC{j+2} ({var_ratio_alpha[j+1]:.1%} variance)')
# plot_scatter(ax2, z_umap, 
#              'B. UMAP Visualization\n(Spliced + Unspliced + Alpha)', 
#              'UMAP 1', 'UMAP 2')

plt.tight_layout()
save_current_figure('Figure3Series_modality_reductions')
print("✓ 绘图完成")

# ===== Original notebook cell 7 =====

import scanpy as sc
import anndata
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scvelo as scv
print("正在运行 Scanpy 标准化流程 (含关键平滑步骤)...")

# ==========================================
# 1. 准备数据
# ==========================================
# 建议：主要使用 Spliced 做流形构建，或者 Spliced + Unspliced
# 这里我们尝试把它们拼起来，利用全部信息
X_s = kinetics_result['spliced']
X_u = kinetics_result['unspliced']

# 转换为 numpy
if hasattr(X_s, 'numpy'): X_s = X_s
if hasattr(X_u, 'numpy'): X_u = X_u

# 确保数据维度对齐
min_len = min(X_s.shape[0], X_u.shape[0])
X_s = X_s[:min_len]
X_u = X_u[:min_len]

# 构建 AnnData，使用 spliced 作为主数据矩阵
adata = sc.AnnData(X=X_s)

# 将 spliced 和 unspliced 数据分别存储到 layers 中
adata.layers['spliced'] = X_s
adata.layers['unspliced'] = X_u

# ==========================================
# 2. 注入几何标签 (为了画图)
# ==========================================
# 获取坐标并打标签
z_traj = topology_data['z_trajectory'].reshape(-1, 2)
# 简单的采样对齐 (如果需要)
if adata.n_obs > 10000:
    indices = np.random.choice(adata.n_obs, 10000, replace=False)
    adata = adata[indices].copy()
    z_sub = z_traj[indices]
else:
    z_sub = z_traj

# 根据坐标打标签
dist = np.linalg.norm(z_sub, axis=1)
# 这里的阈值 1.3 请根据你的 cycle radius 调整
on_cycle = (dist < 1.1) & (dist > 0.9) 
on_edge0 = (z_sub[:, 0] < -1.0) & (~on_cycle)
on_edge1 = (~on_cycle) & (~on_edge0)

labels = np.full(len(z_sub), 'Cycle')
labels[on_edge0] = 'Edge0'
labels[on_edge1] = 'Edge1'
adata.obs['Region'] = labels

# ==========================================
# 3. 关键处理流程
# ==========================================

# A. 基础预处理
sc.pp.log1p(adata)
sc.pp.scale(adata)  # Z-score 标准化，平衡基因权重

# B. 计算 PCA
sc.tl.pca(adata, svd_solver='arpack')

# C. 计算邻居图 (kNN Graph)
# n_neighbors 越大，全局连通性越好
sc.pp.neighbors(adata, n_neighbors=50, n_pcs=30)

# D. 【至关重要的一步】计算矩 (Moments / Smoothing)
# 这一步会利用邻居图对数据进行平滑，填补断层！
print("正在计算 Moments (平滑)...")
scv.pp.moments(adata, n_neighbors=30)

# E. 再次计算降维 (这次基于平滑后的数据)
# 注意：Scanpy 的 moments 会把平滑数据存在 layers 或自动用于后续计算
# 但为了确保 UMAP 用到平滑数据，通常 PCA -> Neighbors -> UMAP 就够了
# 也可以基于 X_pca 跑 UMAP，因为 PCA 也是一种去噪

print("计算 UMAP...")
sc.tl.umap(adata, min_dist=0.5)



# ===== Original notebook cell 8 =====



# ==========================================
# 2. 注入几何标签 (为了画图)
# ==========================================

# ==========================================
# 4. 绘图
# ==========================================
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# PCA
sc.pl.pca(adata, color='Region', ax=axs[0], show=False, title='PCA (Smoothed)')

# UMAP
sc.pl.umap(adata, color='Region', ax=axs[1], show=False, title='UMAP (Smoothed)')

plt.tight_layout()
save_current_figure('Figure3Series_spatial_embedding')

# ===== Original notebook cell 9 =====

adata

# ===== Original notebook cell 10 =====

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.transforms as transforms

# ==========================================
# 1. 数据生成 (保持你的逻辑不变)
# ==========================================


cell_idx = 5
gene_idx = 2
# ODE 求解
time_points_num = topology_data['time_points'].shape[0]
gene_num = kinetics_result['alpha'].shape[1]

t_series = range(time_points_num)

alpha_t = kinetics_result['alpha'].reshape(time_points_num,-1,gene_num)[:,cell_idx,gene_idx]
u_t = kinetics_result['unspliced'].reshape(time_points_num,-1,gene_num)[:,cell_idx,gene_idx]
s_t = kinetics_result['spliced'].reshape(time_points_num,-1,gene_num)[:,cell_idx,gene_idx]


# 为了美观，我们只找第一个大峰值（避免边缘效应）
search_range = slice(0, 100) 
u_peak_idx = np.argmax(u_t[search_range]) 
s_peak_idx = np.argmax(s_t[search_range]) 

u_peak_time = t_series[u_peak_idx]
s_peak_time = t_series[s_peak_idx]
u_peak_val = u_t[u_peak_idx]
s_peak_val = s_t[s_peak_idx]
delta_t = s_peak_time - u_peak_time

# ==========================================
# 2. 绘图与美化
# ==========================================
# 使用稍微宽一点的图，DPI设高
fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

# 颜色定义 (科研常用配色)
col_alpha = '#88CA5E' # 柔和的绿
col_u = '#2E86AB'     # 你的蓝
col_s = '#A23B72'     # 你的紫红

# --- A. 绘制曲线 ---
# Alpha: 使用虚线+填充，表示它是"驱动力"
ax.plot(t_series, alpha_t, color=col_alpha, linestyle='--', linewidth=1.5, label=r'Transcription $\alpha(t)$', alpha=0.8)
ax.fill_between(t_series, 0, alpha_t, color=col_alpha, alpha=0.1) # 淡淡的填充

# U(t): 实线
ax.plot(t_series, u_t, color=col_u, linewidth=2.5, label=r'Unspliced $u(t)$')

# S(t): 实线，稍微粗一点，因为它是最终观测值
ax.plot(t_series, s_t, color=col_s, linewidth=2.5, label=r'Spliced $s(t)$')

# --- B. 强调滞后 (Highlight Delay) ---
# 1. 垂直线 (Drop lines)
# ax.vlines(u_peak_time, 0, u_peak_val, color=col_u, linestyle=':', linewidth=1.5, alpha=0.6)
# ax.vlines(s_peak_time, 0, s_peak_val, color=col_s, linestyle=':', linewidth=1.5, alpha=0.6)

# 2. 峰值点 (Points)
ax.scatter([u_peak_time], [u_peak_val], color='white', edgecolor=col_u, s=60, zorder=10, linewidth=2)
ax.scatter([s_peak_time], [s_peak_val], color='white', edgecolor=col_s, s=60, zorder=10, linewidth=2)

# 3. 阴影区域 (Shaded Band) - 这一步能显著提升图表质感
# 在两个峰值时间之间画一个灰色带
ax.axvspan(u_peak_time, s_peak_time, color='gray', alpha=0.1, zorder=0)

# # --- C. 标注箭头与文字 ---
# arrow_y = max(u_peak_val, s_peak_val) * 1.05

# # 使用 FancyArrowPatch 画双向箭头
# arrow = FancyArrowPatch(
#     (u_peak_time, arrow_y), (s_peak_time, arrow_y),
#     arrowstyle='<->,head_width=4,head_length=6', 
#     linewidth=1.5, color='#D62828', mutation_scale=15
# )
# ax.add_patch(arrow)

# # 文字标注
# ax.text((u_peak_time + s_peak_time) / 2, arrow_y + 0.1, 
#         f'Delay\n$\\Delta t = {delta_t:.2f}$', 
#         ha='center', va='bottom', fontsize=10, fontweight='bold', color='#D62828')

# --- D. 布局美化 (Beautification) ---

# 1. 移除上方和右侧的边框 (Spines)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# 加粗左侧和底部边框
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# 2. 优化网格
ax.grid(True, linestyle='--', alpha=0.3, zorder=0)

# 3. 标签与标题
ax.set_xlabel('Latent Time (t)', fontsize=12, labelpad=8)
ax.set_ylabel('Expression Level', fontsize=12, labelpad=8)
ax.set_title('Kinetic Hysteresis: Molecular Time Lag', fontsize=14, fontweight='bold', pad=15)

# 4. 图例 (去掉边框)
ax.legend(frameon=False, loc='upper right', fontsize=10)

# 5. 调整坐标轴范围 (留一点呼吸感)
# ax.set_ylim(0, max(alpha_t.max(), s_t.max()) * 1.3)
# ax.set_xlim(0, 10)

plt.tight_layout()
save_current_figure('Figure3Series_time_lag')

# ===== Original notebook cell 11 =====

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LinearSegmentedColormap

# ==========================================
# 0. 准备工作 (模拟环境，保持你的数据读取逻辑)
# ==========================================
# 假设 kinetics_result 和 config_para 已经存在
gene_idx = 20 
config_para = yaml.load(open(DATA_DIR / "auto_topology_config.yaml", "r"),Loader=yaml.FullLoader)
beta = config_para['genes'][gene_idx]['beta']
gamma = config_para['genes'][gene_idx]['gamma']

# 获取数据
u_data = kinetics_result['unspliced'][:, gene_idx]
s_data = kinetics_result['spliced'][:, gene_idx]
alpha_vals = kinetics_result['alpha'][:, gene_idx] # 用 alpha 来上色！

# ==========================================
# 1. 设置画布风格
# ==========================================
fig, ax_c = plt.subplots(figsize=(6, 6), dpi=120)

# 定义一个漂亮的 colormap (从灰色到 teal/紫色)
# 这样低 alpha (抑制期) 是灰色，高 alpha (诱导期) 是亮色
colors = ["#E0E0E0", "#48A9A6", "#4281A4", "#2E5077"]
cmap_custom = LinearSegmentedColormap.from_list("custom_velocity", colors)

# ==========================================
# 2. 绘制散点 (Phase Portrait)
# ==========================================
# 根据 alpha 值上色，这能完美区分上半弧和下半弧
sc = ax_c.scatter(s_data, u_data, c=alpha_vals, cmap=cmap_custom, 
                  alpha=0.8, s=20, edgecolors='none', zorder=5)

# ==========================================
# 3. 绘制理论稳态线 (Steady State Line)
# ==========================================
# 修正稳态公式: ds/dt = beta*u - gamma*s = 0 => u = (gamma/beta) * s
# 注意：如果你的模型里 s 是 counts，u 也是 counts，斜率就是 gamma/beta
# 如果你之前的代码是用 beta/gamma，请确认你的 beta/gamma 定义是否反了
# 这里我按照标准公式 u = (gamma/beta)*s 写，如果不重合请换回你的写法
slope = gamma / beta 
max_s = s_data.max() * 1.1
line_x = np.linspace(0, max_s, 100)
line_y = slope * line_x

ax_c.plot(line_x, line_y, color='black', linestyle='--', linewidth=1.5, 
          alpha=0.7, label='Steady State\n($u = \\frac{\\gamma}{\\beta} s$)', zorder=2)

# ==========================================
# 4. 标注与修饰 (Annotations)
# ==========================================

# A. 标注 Indution / Repression 区域
# 找两个代表性的点来放文字
# ax_c.text(max_s * 0.3, max_s * slope * 1.5, 'Induction\n(Genes On)', 
#           fontsize=10, color='#2E5077', fontweight='bold', ha='center')
# ax_c.text(max_s * 0.7, max_s * slope * 0.5, 'Repression\n(Genes Off)', 
#           fontsize=10, color='gray', fontweight='bold', ha='center')

# B. 绘制环形方向箭头 (Curved Arrow)
# 在环的中心画一个逆时针箭头
# from matplotlib.patches import FancyArrowPatch
# arrow = FancyArrowPatch((max_s*0.6, max_s*slope*0.9), (max_s*0.4, max_s*slope*1.1),
#                         connectionstyle="arc3,rad=.3", arrowstyle='simple,head_width=8,head_length=8',
#                         color='k', alpha=0.5, zorder=10)
# ax_c.add_patch(arrow)

# C. 坐标轴与标题
ax_c.set_xlabel('Spliced Expression ($s$)', fontsize=12, labelpad=10)
ax_c.set_ylabel('Unspliced Expression ($u$)', fontsize=12, labelpad=10)
ax_c.set_title(f'Gene {gene_idx}: Phase Portrait', fontsize=14, fontweight='bold', pad=15)

# D. 边框处理 (Despining)
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)
ax_c.spines['left'].set_linewidth(1.2)
ax_c.spines['bottom'].set_linewidth(1.2)

# E. 颜色条 (Colorbar) - 可选
# cbar = plt.colorbar(sc, ax=ax_c, fraction=0.046, pad=0.04)
# cbar.set_label(r'Transcription Rate ($\alpha$)', fontsize=10)
# cbar.outline.set_visible(False)

# F. 网格
ax_c.grid(True, linestyle=':', alpha=0.3, zorder=0)
ax_c.legend(frameon=False, loc='upper right')

print("✓ Panel C 美化完成")
plt.tight_layout()
save_current_figure('Figure3Series_phase_portrait')

# ===== Original notebook cell 12 =====

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

print("绘制 Panel D (网格向量场版)...")

fig, ax_d = plt.subplots(figsize=(6, 6), dpi=120)

# ==========================================
# 1. 数据准备
# ==========================================
u_data = kinetics_result['unspliced'][:, gene_idx]
s_data = kinetics_result['spliced'][:, gene_idx]
u_local = u_data if isinstance(u_data, np.ndarray) else np.array(u_data)
s_local = s_data if isinstance(s_data, np.ndarray) else np.array(s_data)

# 计算真实速度 (Ground Truth Velocity)
alpha_vals = kinetics_result['alpha'][:, gene_idx]
alpha_local = alpha_vals if isinstance(alpha_vals, np.ndarray) else np.array(alpha_vals)
beta = synth.beta[gene_idx].item() if isinstance(synth.beta, np.ndarray) else synth.beta[gene_idx]
gamma = synth.gamma[gene_idx].item() if isinstance(synth.gamma, np.ndarray) else synth.gamma[gene_idx]

v_s = beta * u_local - gamma * s_local
v_u = alpha_local - beta * u_local

# ==========================================
# 2. 核心技术 A: KNN 平滑流形 (Smoothed Manifold)
# ==========================================
# 原理：每个点的坐标替换为其 k 个邻居的中心。这能极大地收缩点云宽度，显露骨架。
# 采样一部分点进行平滑，避免计算量过大
n_samples = 1000
if len(s_local) > n_samples:
    idx = np.random.choice(len(s_local), n_samples, replace=False)
    s_sub, u_sub = s_local[idx], u_local[idx]
else:
    s_sub, u_sub = s_local, u_local

X = np.vstack((s_sub, u_sub)).T
# k越大，线条越细越滑，但也可能过度收缩
nbrs = NearestNeighbors(n_neighbors=50, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)

# 计算邻居均值作为新的骨架点
s_smooth = np.mean(s_sub[indices], axis=1)
u_smooth = np.mean(u_sub[indices], axis=1)

# 为了画出连续的线，我们需要对这些平滑后的点排序 (按角度)
# 这一步仅为了连线视觉效果，如果不连线，光画点也可以形成一条“线”
s_center, u_center = np.mean(s_smooth), np.mean(u_smooth)
angles = np.arctan2(u_smooth - u_center, s_smooth - s_center)
sort_order = np.argsort(angles)
s_smooth_sorted = s_smooth[sort_order]
u_smooth_sorted = u_smooth[sort_order]

# ==========================================
# 3. 核心技术 B: 网格化向量场 (Grid Quiver)
# ==========================================
# 定义网格
n_grid = 12 # 12x12 的网格
s_min, s_max = s_local.min(), s_local.max()
u_min, u_max = u_local.min(), u_local.max()
# 留一点边距
margin = 0.05
s_grid = np.linspace(s_min, s_max, n_grid)
u_grid = np.linspace(u_min, u_max, n_grid)

# 存储网格箭头数据
s_arrows, u_arrows = [], []
vs_arrows, vu_arrows = [], []

# 对每个网格点，寻找附近的真实数据，计算平均速度
grid_radius = (s_max - s_min) / n_grid * 1.5 # 搜索半径
all_points = np.vstack((s_local, u_local)).T
tree = NearestNeighbors(radius=grid_radius).fit(all_points)

for si in s_grid:
    for ui in u_grid:
        # 找到网格点半径内的所有真实细胞
        # query_radius 返回的是索引数组的数组
        indices = tree.radius_neighbors([[si, ui]], return_distance=False)[0]
        
        if len(indices) > 10: # 只有当附近有足够多的细胞时才画箭头
            # 计算局部平均位置（让箭头根部吸附在数据中心，而不是死板的网格点上）
            s_mean = np.mean(s_local[indices])
            u_mean = np.mean(u_local[indices])
            
            # 计算局部平均速度
            vs_mean = np.mean(v_s[indices])
            vu_mean = np.mean(v_u[indices])
            
            s_arrows.append(s_mean)
            u_arrows.append(u_mean)
            vs_arrows.append(vs_mean)
            vu_arrows.append(vu_mean)

# ==========================================
# 4. 绘图层
# ==========================================

# A. 背景噪声云 (Raw Data)
ax_d.scatter(s_local, u_local, c='#9CA3AF', alpha=0.15, s=10, edgecolors='none', label='Observations')

# B. 平滑流形骨架 (KNN Smoothed)
# 用散点表示骨架比连线更稳健，不会出现“飞线”
ax_d.scatter(s_smooth, u_smooth, c='#1E3A8A', s=15, alpha=0.6, label='Smoothed Manifold', zorder=2)
# 如果你非常想要连线，可以用下面的代码（由于排序可能不完美，可能需要在视觉上取舍）
# ax_d.plot(s_smooth_sorted, u_smooth_sorted, c='#1E3A8A', lw=2, alpha=0.5)

# C. 速度向量 (Grid Arrows)
# 归一化长度，保证箭头大小一致，只展示方向
v_norms = np.sqrt(np.array(vs_arrows)**2 + np.array(vu_arrows)**2)
# 避免除零
v_norms[v_norms == 0] = 1e-6
# 统一缩放因子
scale = 0.8
vs_plot = (np.array(vs_arrows) / v_norms) * scale
vu_plot = (np.array(vu_arrows) / v_norms) * scale

ax_d.quiver(s_arrows, u_arrows, vs_arrows, vu_arrows,
            color='#EF4444', 
            angles='xy', scale_units='xy', scale=None, # 让 matplotlib 自动计算最佳比例
            
            alpha=1.0, zorder=10, label='Avg Velocity Field')

# ==========================================
# 5. 美化
# ==========================================
ax_d.set_xlabel('Spliced ($s$)', fontsize=12)
ax_d.set_ylabel('Unspliced ($u$)', fontsize=12)
ax_d.set_title('D. Velocity Tangency Verification\n(Grid-Averaged Flow)', fontsize=13, fontweight='bold')
ax_d.set_aspect('equal')
ax_d.spines['top'].set_visible(False)
ax_d.spines['right'].set_visible(False)
ax_d.grid(True, linestyle=':', alpha=0.4)
# 调整图例位置
ax_d.legend(frameon=False, loc='upper left', bbox_to_anchor=(1, 1))
ax_d.set_xlim(-0.1, 0.8)
ax_d.set_ylim(-0.1, 0.8)
plt.tight_layout()
print("✓ 网格向量场版 Panel D 完成")
save_current_figure('Figure3Series_vector_field')

# ===== Original notebook cell 13 =====

# Panel E: Parameter Control (γ/β Ratio)
print("绘制Panel E: Parameter Control...")

# 创建独立的figure
fig_e, ax_e = plt.subplots(figsize=(6, 6))

# 从YAML配置文件中读取真实的beta和gamma值
import yaml
config_path = DATA_DIR / "auto_topology_config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config_dict = yaml.safe_load(f)

# 选择几个不同gamma/beta比值的cycle基因
# 计算所有cycle基因的gamma/beta比值
cycle_genes = []
for gene in config_dict['genes']:
    if gene['type'] == 'cycle':
        beta = gene.get('beta', 1.0)
        gamma = gene.get('gamma', 1.0)
        ratio = gamma / beta if beta > 0 else 0
        cycle_genes.append({
            'name': gene.get('name', 'unknown'),
            'beta': beta,
            'gamma': gamma,
            'ratio': ratio,
            'gene_idx': None  # 稍后填充
        })

# 按ratio排序，选择低、中、高三个比值
cycle_genes.sort(key=lambda x: x['ratio'])
n_genes = len(cycle_genes)
if n_genes >= 3:
    selected_indices = [0, n_genes // 2, n_genes - 1]  # 低、中、高
else:
    selected_indices = list(range(n_genes))

# 找到这些基因在synth中的索引
gene_name_to_idx = {}
for i, name in enumerate(synth.gene_names):
    if name:
        gene_name_to_idx[name] = i

# 为选中的基因填充索引
for gene_info in cycle_genes:
    gene_info['gene_idx'] = gene_name_to_idx.get(gene_info['name'])

# 选择三个不同比值的基因
selected_genes = [cycle_genes[i] for i in selected_indices[:3]]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
labels = ['Low γ/β (Fat)', 'Medium γ/β', 'High γ/β (Thin)']

# 从真实数据中提取极限环
# 对于每个选中的基因，从kinetics_result中提取u和s数据
# 选择一个在cycle上的细胞，提取其完整轨迹
n_cells = topology_data['z_data'].shape[0]
n_steps = z_trajectory.shape[0]

# 选择一个在cycle上的细胞（距离原点约1.0的细胞）
z_final = topology_data['z_data']
dist_to_origin = np.linalg.norm(z_final, axis=1)
cycle_cell_idx = np.argmin(np.abs(dist_to_origin - 1.0))  # 选择最接近cycle rim的细胞

# 提取这个细胞的完整轨迹
for i, gene_info in enumerate(selected_genes):
    if gene_info['gene_idx'] is None:
        continue
    
    gene_idx = gene_info['gene_idx']
    beta = gene_info['beta']
    gamma = gene_info['gamma']
    ratio = gene_info['ratio']
    color = colors[i]
    label = f"{labels[i]}\n(γ/β={ratio:.2f})"
    
    # 提取这个细胞在整个轨迹上的u和s数据
    # kinetics_result的形状是 [n_steps * n_cells, n_genes]
    # 需要提取特定细胞在所有时间步的数据
    u_traj = []
    s_traj = []
    
    for step in range(n_steps):
        cell_idx_flat = step * n_cells + cycle_cell_idx
        if cell_idx_flat < kinetics_result['unspliced'].shape[0]:
            u_val = kinetics_result['unspliced'][cell_idx_flat, gene_idx]
            s_val = kinetics_result['spliced'][cell_idx_flat, gene_idx]
            # 转换为numpy
            if isinstance(u_val, np.ndarray):
                u_val = u_val.item()
            if isinstance(s_val, np.ndarray):
                s_val = s_val.item()
            u_traj.append(u_val)
            s_traj.append(s_val)
    
    if len(u_traj) > 0:
        u_traj = np.array(u_traj)
        s_traj = np.array(s_traj)
        # 绘制极限环
        ax_e.plot(s_traj, u_traj, '-', color=color, linewidth=2.5, label=label, alpha=0.8)

ax_e.set_xlabel('Spliced (s)', fontsize=11)
ax_e.set_ylabel('Unspliced (u)', fontsize=11)
ax_e.set_title('E. Parameter Control (γ/β)', fontsize=12, fontweight='bold')
ax_e.legend(loc='best', fontsize=9)
ax_e.grid(True, alpha=0.3)
ax_e.set_aspect('equal', adjustable='box')

print("✓ Panel E完成")
save_current_figure("Figure3Series_parameter_control")

# ===== Original notebook cell 14 =====

# Panel F: Stochasticity vs. Deterministic (噪声效应)
print("绘制Panel F: Stochasticity vs. Deterministic...")

# 创建独立的figure
fig_f, ax_f = plt.subplots(figsize=(6, 6))

# 选择一个cycle基因用于展示
# 从YAML配置中选择一个cycle基因
import yaml
config_path = DATA_DIR / "auto_topology_config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config_dict = yaml.safe_load(f)

# 选择第一个cycle基因
cycle_gene = None
for gene in config_dict['genes']:
    if gene['type'] == 'cycle':
        cycle_gene = gene
        break

if cycle_gene is None:
    print("警告: 未找到cycle基因")
else:
    gene_name = cycle_gene['name']
    beta = cycle_gene.get('beta', 1.0)
    gamma = cycle_gene.get('gamma', 1.0)
    
    # 找到基因索引
    gene_name_to_idx = {}
    for i, name in enumerate(synth.gene_names):
        if name:
            gene_name_to_idx[name] = i
    
    gene_idx = gene_name_to_idx.get(gene_name)
    
    if gene_idx is None:
        print(f"警告: 未找到基因 {gene_name} 的索引")
    else:
        # 选择一个在cycle上的细胞
        n_cells = topology_data['z_data'].shape[0]
        n_steps = z_trajectory.shape[0]
        z_final = topology_data['z_data']
        dist_to_origin = np.linalg.norm(z_final, axis=1)
        cycle_cell_idx = np.argmin(np.abs(dist_to_origin - 1.0))
        
        # 提取确定性解（无噪声的连续数据）
        # kinetics_result是simulate_counts=False生成的，所以是连续值
        u_det = []
        s_det = []
        
        for step in range(n_steps):
            cell_idx_flat = step * n_cells + cycle_cell_idx
            if cell_idx_flat < kinetics_result['unspliced'].shape[0]:
                u_val = kinetics_result['unspliced'][cell_idx_flat, gene_idx]
                s_val = kinetics_result['spliced'][cell_idx_flat, gene_idx]
                # 转换为numpy
                if isinstance(u_val, np.ndarray):
                    u_val = u_val.item()
                if isinstance(s_val, np.ndarray):
                    s_val = s_val.item()
                u_det.append(u_val)
                s_det.append(s_val)
        
        if len(u_det) > 0:
            u_det = np.array(u_det)
            s_det = np.array(s_det)
            
            # 绘制确定性解（平滑的极限环）
            ax_f.plot(s_det, u_det, 'k-', linewidth=3, label='Deterministic', alpha=0.9, zorder=10)
            
            # 如果有带噪声的数据（simulate_counts=True），可以对比
            # 这里我们使用相同的轨迹数据，但添加不同水平的噪声来模拟随机性
            # 或者，如果之前生成了simulate_counts=True的数据，可以使用那个
            
            # 方法1: 使用真实数据添加不同噪声水平
            noise_levels = [0.05, 0.15, 0.25]
            colors_noise = ['#95E1D3', '#F38181', '#AA96DA']
            
            for noise_level, color in zip(noise_levels, colors_noise):
                # 添加高斯噪声
                u_noise = u_det + np.random.normal(0, noise_level * u_det.std(), size=u_det.shape)
                s_noise = s_det + np.random.normal(0, noise_level * s_det.std(), size=s_det.shape)
                # 确保非负
                u_noise = np.clip(u_noise, 0, None)
                s_noise = np.clip(s_noise, 0, None)
                
                # 稀疏采样用于可视化
                sample_indices = np.linspace(0, len(u_noise)-1, min(50, len(u_noise)), dtype=int)
                ax_f.scatter(s_noise[sample_indices], u_noise[sample_indices], 
                           c=color, alpha=0.5, s=15, edgecolors='none', 
                           label=f'Noise σ={noise_level:.2f}', zorder=5)
            
            ax_f.set_xlabel('Spliced (s)', fontsize=11)
            ax_f.set_ylabel('Unspliced (u)', fontsize=11)
            ax_f.set_title('F. Stochasticity vs. Deterministic', fontsize=12, fontweight='bold')
            ax_f.legend(loc='best', fontsize=9)
            ax_f.grid(True, alpha=0.3)
            ax_f.set_aspect('equal', adjustable='box')

print("✓ Panel F完成")
save_current_figure("Figure3Series_stochasticity")

# ===== Original notebook cell 15 =====

adata

# ===== Original notebook cell 17 =====

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from matplotlib import gridspec

# --- 绘图 ---
fig = plt.figure(figsize=(18, 15.5))
gs = gridspec.GridSpec(
    3,
    4,
    figure=fig,
    width_ratios=[1, 1, 1, 1.5],
    height_ratios=[1, 1, 0.62],
    hspace=0.5,
    wspace=0.35,
)

big_fig_sample_size = 3000
z_trajectory_sample_indices = np.random.randint(
    0,
    z_trajectory.shape[0] * z_trajectory.shape[1],
    size=big_fig_sample_size,
)

# ==========================================
# 1. 数据提取 (Extract)
# ==========================================

# A. 提取三个矩阵
X_s = kinetics_result['spliced']

if hasattr(X_s, 'numpy'): X_s = X_s


# 确保形状一致 (N_samples, n_genes)
X_s = X_s.reshape(-1, X_s.shape[1]) 

# B. 获取几何坐标
z_traj = topology_data['z_trajectory']
z_flat = z_traj.reshape(-1, 2)

# C. 对齐行数 (以防万一)
min_len = min(X_s.shape[0], z_flat.shape[0])
X_s = X_s[:min_len]
z_flat = z_flat[:min_len]


# 使用相同索引采样所有模态
X_s_sub = X_s[z_trajectory_sample_indices]
z_sub = z_flat[z_trajectory_sample_indices]

print(f"采样完成: {X_s_sub.shape}")

# ==========================================
# 3. 独立预处理 (Independent Preprocessing)
# ==========================================
# 这一步是关键：分别处理，消除不同物理量的量纲差异

def preprocess_modality(X_data, name):
    """辅助函数：对单一模态进行 Log1p 和 StandardScaler"""
    # 1. Log1p: 处理偏态分布，压制极大值
    X_log = np.log1p(X_data)
    
    # 2. Scale: 转换为 Z-score (均值0，方差1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)
    print(f" -> {name} Standardized. Mean~{X_scaled.mean():.2f}, Std~{X_scaled.std():.2f}")
    return X_scaled

print("开始独立标准化...")
X_s_scaled = preprocess_modality(X_s_sub, "Spliced")

# ==========================================
# 4. 特征融合 (Fusion)
# ==========================================

# ==========================================
# 5. 定义颜色标签
# ==========================================
dist_to_origin = np.linalg.norm(z_sub, axis=1)
# 稍微放宽范围防止断裂
on_cycle = (dist_to_origin < 1.35) & (dist_to_origin > 0.65) 
on_edge0 = (z_sub[:, 0] < -1.0) & (~on_cycle)
on_edge1 = (~on_cycle) & (~on_edge0)

point_labels = np.zeros(len(z_sub), dtype=int)
point_labels[on_edge0] = 0
point_labels[on_cycle] = 1
point_labels[on_edge1] = 2

colors_map = ['#2E86AB', '#A23B72', '#F18F01']
label_names = ['Edge0 (Start)', 'Cycle', 'Edge1 (End)']

# ==========================================
# 6. 降维与绘图
# ==========================================

# --- PCA ---
print("计算 PCA...")
# 使用融合后的数据
pca_spliced = PCA(n_components=4, random_state=42)
z_pca_spliced = pca_spliced.fit_transform(X_s_scaled) 
var_ratio_spliced = pca_spliced.explained_variance_ratio_


pca_unspliced = PCA(n_components=4, random_state=42)
z_pca_unspliced = pca_unspliced.fit_transform(X_u_scaled) 
var_ratio_unspliced = pca_unspliced.explained_variance_ratio_

pca_alpha = PCA(n_components=4, random_state=42)
z_pca_alpha = pca_alpha.fit_transform(X_a_scaled) 
var_ratio_alpha = pca_alpha.explained_variance_ratio_




gene_idx =16

# 第一个子图：alpha
ax = fig.add_subplot(gs[0,0])
color1 = kinetics_result['alpha'][z_trajectory_sample_indices,gene_idx]
scatter1 = ax.scatter(z_trajectory.reshape(-1,2)[z_trajectory_sample_indices,0],
                        z_trajectory.reshape(-1,2)[z_trajectory_sample_indices,1],
                        s=10,alpha=0.5,c=color1,cmap='TopoVelo_grad', rasterized=True)
ax.set_title('alpha of cycle gene 5')
# ax.set_aspect('equal')
ax.axis('off')
plt.colorbar(scatter1, ax=ax, label='Alpha (transcription rate)')

# 第二个子图：unspliced
color2 = kinetics_result['unspliced'][z_trajectory_sample_indices,gene_idx]
ax = fig.add_subplot(gs[0,1])
scatter2 = ax.scatter(z_trajectory.reshape(-1,2)[z_trajectory_sample_indices,0],
                        z_trajectory.reshape(-1,2)[z_trajectory_sample_indices,1],
                        s=10,alpha=0.5,c=color2,cmap='TopoVelo_grad', rasterized=True)
ax.set_title('unspliced RNA of cycle gene 5')
# ax.set_aspect('equal')
ax.axis('off')
plt.colorbar(scatter2, ax=ax, label='Unspliced RNA')

# 第三个子图：spliced
color3 = kinetics_result['spliced'][z_trajectory_sample_indices,gene_idx]
ax = fig.add_subplot(gs[0,2])
scatter3 = ax.scatter(z_trajectory.reshape(-1,2)[z_trajectory_sample_indices,0],
                        z_trajectory.reshape(-1,2)[z_trajectory_sample_indices,1],
                        s=10,alpha=0.5,c=color3,cmap='TopoVelo_grad', rasterized=True)
ax.set_title('spliced RNA of cycle gene 5')
# ax.set_aspect('equal')
ax.axis('off')
plt.colorbar(scatter3, ax=ax, label='Spliced RNA')


def plot_scatter(ax, coords, title, xlabel, ylabel):
    for i, color in enumerate(colors_map):
        mask = point_labels == i
        if np.sum(mask) > 0:
            sub_coords = coords[mask]
            ax.scatter(sub_coords[:, 0], sub_coords[:, 1], 
                      c=color, label=label_names[i], 
                      s=10, alpha=0.6, edgecolors='none', rasterized=True)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=10, markerscale=2)
    ax.grid(True, alpha=0.2, linestyle='--')


for j in range(3):
    ax = fig.add_subplot(gs[1,j])
    plot_scatter(ax, z_pca_spliced[:,[j,j+1]], 
        f'A. PCA Visualization\n(Spliced PC{j+1}+PC{j+2})', 
        f'PC{j+1} ({var_ratio_spliced[j]:.1%} variance)', 
        f'PC{j+2} ({var_ratio_spliced[j+1]:.1%} variance)')


# plt.tight_layout()
# save_current_figure('Figure3Series_combined_panels')
# print("✓ 绘图完成")


# ==========================================
# 1. 数据生成 (保持你的逻辑不变)
# ==========================================
sub_gs = gs[0:2, 3].subgridspec(3, 1, hspace=0.2)  

def plot_kinetic_hysteresis(ax, kinetics_result, cell_idx, gene_idx):
    # ODE 求解
    time_points_num = topology_data['time_points'].shape[0]
    gene_num = kinetics_result['alpha'].shape[1]

    t_series = range(time_points_num)

    alpha_t = kinetics_result['alpha'].reshape(time_points_num,-1,gene_num)[:,cell_idx,gene_idx]
    u_t = kinetics_result['unspliced'].reshape(time_points_num,-1,gene_num)[:,cell_idx,gene_idx]
    s_t = kinetics_result['spliced'].reshape(time_points_num,-1,gene_num)[:,cell_idx,gene_idx]


    # 为了美观，我们只找第一个大峰值（避免边缘效应）
    search_range = slice(0, 100) 
    u_peak_idx = np.argmax(u_t[search_range]) 
    s_peak_idx = np.argmax(s_t[search_range]) 

    u_peak_time = t_series[u_peak_idx]
    s_peak_time = t_series[s_peak_idx]
    u_peak_val = u_t[u_peak_idx]
    s_peak_val = s_t[s_peak_idx]
    delta_t = s_peak_time - u_peak_time

    # ==========================================
    # 2. 绘图与美化
    # ==========================================

    # 颜色定义 (科研常用配色)
    col_alpha = '#88CA5E' # 柔和的绿
    col_u = '#2E86AB'     # 你的蓝
    col_s = '#A23B72'     # 你的紫红

    # --- A. 绘制曲线 ---
    # Alpha: 使用虚线+填充，表示它是"驱动力"
    ax.plot(t_series, alpha_t, color=col_alpha, linestyle='--', linewidth=1.5, label=r'Transcription $\alpha(t)$', alpha=0.8)
    ax.fill_between(t_series, 0, alpha_t, color=col_alpha, alpha=0.1) # 淡淡的填充

    # U(t): 实线
    ax.plot(t_series, u_t, color=col_u, linewidth=2.5, label=r'Unspliced $u(t)$')

    # S(t): 实线，稍微粗一点，因为它是最终观测值
    ax.plot(t_series, s_t, color=col_s, linewidth=2.5, label=r'Spliced $s(t)$')

    # --- B. 强调滞后 (Highlight Delay) ---
    # 1. 垂直线 (Drop lines)
    # ax.vlines(u_peak_time, 0, u_peak_val, color=col_u, linestyle=':', linewidth=1.5, alpha=0.6)
    # ax.vlines(s_peak_time, 0, s_peak_val, color=col_s, linestyle=':', linewidth=1.5, alpha=0.6)

    # 2. 峰值点 (Points)
    ax.scatter([u_peak_time], [u_peak_val], color='white', edgecolor=col_u, s=60, zorder=10, linewidth=2)
    ax.scatter([s_peak_time], [s_peak_val], color='white', edgecolor=col_s, s=60, zorder=10, linewidth=2)

    # 3. 阴影区域 (Shaded Band) - 这一步能显著提升图表质感
    # 在两个峰值时间之间画一个灰色带
    ax.axvspan(u_peak_time, s_peak_time, color='gray', alpha=0.1, zorder=0)


    # --- D. 布局美化 (Beautification) ---

    # 1. 移除上方和右侧的边框 (Spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 加粗左侧和底部边框
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    # 2. 优化网格
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)

    # 3. 标签与标题
    ax.set_xlabel('Latent Time (t)', fontsize=12, labelpad=8)
    ax.set_ylabel('Expression Level', fontsize=12, labelpad=8)
    ax.set_title('Kinetic Hysteresis: Molecular Time Lag', fontsize=14, fontweight='bold', pad=15)

    # 4. 图例 (去掉边框)
    ax.legend(frameon=False, loc='upper right', fontsize=10)


ax = fig.add_subplot(sub_gs[0])
plot_kinetic_hysteresis(ax, kinetics_result, 100, 2)
ax = fig.add_subplot(sub_gs[1])
plot_kinetic_hysteresis(ax, kinetics_result, 100, 35)
ax = fig.add_subplot(sub_gs[2])
plot_kinetic_hysteresis(ax, kinetics_result, 100, 45)


bottom_axes = [fig.add_subplot(gs[2, i]) for i in range(4)]

# G. Phase portrait
ax_g = bottom_axes[0]
phase_gene_idx = 20
config_para = yaml.load(open(DATA_DIR / "auto_topology_config.yaml", "r"), Loader=yaml.FullLoader)
beta = config_para["genes"][phase_gene_idx]["beta"]
gamma = config_para["genes"][phase_gene_idx]["gamma"]
u_data = kinetics_result["unspliced"][:, phase_gene_idx]
s_data = kinetics_result["spliced"][:, phase_gene_idx]
alpha_vals = kinetics_result["alpha"][:, phase_gene_idx]
cmap_phase = LinearSegmentedColormap.from_list(
    "custom_velocity_big",
    ["#E0E0E0", "#48A9A6", "#4281A4", "#2E5077"],
)
ax_g.scatter(
    s_data,
    u_data,
    c=alpha_vals,
    cmap=cmap_phase,
    alpha=0.8,
    s=8,
    edgecolors="none",
    zorder=5,
    rasterized=True,
)
slope = gamma / beta
max_s = s_data.max() * 1.1
line_x = np.linspace(0, max_s, 100)
ax_g.plot(
    line_x,
    slope * line_x,
    color="black",
    linestyle="--",
    linewidth=1.2,
    alpha=0.7,
    label="Steady State",
    zorder=2,
)
ax_g.set_xlabel("Spliced (s)")
ax_g.set_ylabel("Unspliced (u)")
ax_g.set_title("G. Phase Portrait", fontweight="bold")
ax_g.grid(True, linestyle=":", alpha=0.3, zorder=0)
ax_g.legend(frameon=False, fontsize=7)

# H. Velocity tangency verification
ax_h = bottom_axes[1]
u_local = u_data if isinstance(u_data, np.ndarray) else np.array(u_data)
s_local = s_data if isinstance(s_data, np.ndarray) else np.array(s_data)
alpha_local = alpha_vals if isinstance(alpha_vals, np.ndarray) else np.array(alpha_vals)
beta_val = synth.beta[phase_gene_idx].item() if isinstance(synth.beta, np.ndarray) else synth.beta[phase_gene_idx]
gamma_val = synth.gamma[phase_gene_idx].item() if isinstance(synth.gamma, np.ndarray) else synth.gamma[phase_gene_idx]
v_s = beta_val * u_local - gamma_val * s_local
v_u = alpha_local - beta_val * u_local
n_samples = 1000
if len(s_local) > n_samples:
    idx = np.random.choice(len(s_local), n_samples, replace=False)
    s_sub, u_sub = s_local[idx], u_local[idx]
else:
    s_sub, u_sub = s_local, u_local
X_phase = np.vstack((s_sub, u_sub)).T
nbrs = NearestNeighbors(n_neighbors=50, algorithm="ball_tree").fit(X_phase)
_, indices = nbrs.kneighbors(X_phase)
s_smooth = np.mean(s_sub[indices], axis=1)
u_smooth = np.mean(u_sub[indices], axis=1)
n_grid = 12
s_grid = np.linspace(s_local.min(), s_local.max(), n_grid)
u_grid = np.linspace(u_local.min(), u_local.max(), n_grid)
s_arrows, u_arrows, vs_arrows, vu_arrows = [], [], [], []
grid_radius = (s_local.max() - s_local.min()) / n_grid * 1.5
tree = NearestNeighbors(radius=grid_radius).fit(np.vstack((s_local, u_local)).T)
for si in s_grid:
    for ui in u_grid:
        nn = tree.radius_neighbors([[si, ui]], return_distance=False)[0]
        if len(nn) > 10:
            s_arrows.append(np.mean(s_local[nn]))
            u_arrows.append(np.mean(u_local[nn]))
            vs_arrows.append(np.mean(v_s[nn]))
            vu_arrows.append(np.mean(v_u[nn]))
ax_h.scatter(
    s_local,
    u_local,
    c="#9CA3AF",
    alpha=0.12,
    s=6,
    edgecolors="none",
    label="Observations",
    rasterized=True,
)
ax_h.scatter(s_smooth, u_smooth, c="#1E3A8A", s=8, alpha=0.55, label="Smoothed", rasterized=True)
ax_h.quiver(s_arrows, u_arrows, vs_arrows, vu_arrows, color="#EF4444", angles="xy", scale_units="xy", scale=None)
ax_h.set_xlabel("Spliced (s)")
ax_h.set_ylabel("Unspliced (u)")
ax_h.set_title("H. Velocity Tangency", fontweight="bold")
ax_h.grid(True, linestyle=":", alpha=0.3)
ax_h.legend(frameon=False, fontsize=7)

# I. Parameter control
ax_i = bottom_axes[2]
for i, gene_info in enumerate(selected_genes):
    if gene_info["gene_idx"] is None:
        continue
    this_gene_idx = gene_info["gene_idx"]
    ratio = gene_info["ratio"]
    color = colors[i]
    label = f"{labels[i]}\n(γ/β={ratio:.2f})"
    u_traj, s_traj = [], []
    for step in range(n_steps):
        cell_idx_flat = step * n_cells + cycle_cell_idx
        if cell_idx_flat < kinetics_result["unspliced"].shape[0]:
            u_val = kinetics_result["unspliced"][cell_idx_flat, this_gene_idx]
            s_val = kinetics_result["spliced"][cell_idx_flat, this_gene_idx]
            u_traj.append(u_val.item() if isinstance(u_val, np.ndarray) else u_val)
            s_traj.append(s_val.item() if isinstance(s_val, np.ndarray) else s_val)
    if u_traj:
        ax_i.plot(s_traj, u_traj, "-", color=color, linewidth=1.8, label=label, alpha=0.85)
ax_i.set_xlabel("Spliced (s)")
ax_i.set_ylabel("Unspliced (u)")
ax_i.set_title("I. Parameter Control", fontweight="bold")
ax_i.grid(True, alpha=0.3)
ax_i.legend(frameon=False, fontsize=6)

# J. Stochasticity
ax_j = bottom_axes[3]
cycle_gene = next((gene for gene in config_dict["genes"] if gene["type"] == "cycle"), None)
if cycle_gene is not None:
    gene_name = cycle_gene["name"]
    gene_name_to_idx = {name: i for i, name in enumerate(synth.gene_names) if name}
    stochastic_gene_idx = gene_name_to_idx.get(gene_name)
    if stochastic_gene_idx is not None:
        u_det, s_det = [], []
        for step in range(n_steps):
            cell_idx_flat = step * n_cells + cycle_cell_idx
            if cell_idx_flat < kinetics_result["unspliced"].shape[0]:
                u_val = kinetics_result["unspliced"][cell_idx_flat, stochastic_gene_idx]
                s_val = kinetics_result["spliced"][cell_idx_flat, stochastic_gene_idx]
                u_det.append(u_val.item() if isinstance(u_val, np.ndarray) else u_val)
                s_det.append(s_val.item() if isinstance(s_val, np.ndarray) else s_val)
        u_det = np.array(u_det)
        s_det = np.array(s_det)
        ax_j.plot(s_det, u_det, "k-", linewidth=2.0, label="Deterministic", alpha=0.9, zorder=10)
        for noise_level, color in zip([0.05, 0.15, 0.25], ["#95E1D3", "#F38181", "#AA96DA"]):
            u_noise = np.clip(u_det + np.random.normal(0, noise_level * u_det.std(), size=u_det.shape), 0, None)
            s_noise = np.clip(s_det + np.random.normal(0, noise_level * s_det.std(), size=s_det.shape), 0, None)
            sample_indices = np.linspace(0, len(u_noise) - 1, min(50, len(u_noise)), dtype=int)
            ax_j.scatter(
                s_noise[sample_indices],
                u_noise[sample_indices],
                c=color,
                alpha=0.5,
                s=10,
                edgecolors="none",
                label=f"Noise σ={noise_level:.2f}",
                rasterized=True,
            )
ax_j.set_xlabel("Spliced (s)")
ax_j.set_ylabel("Unspliced (u)")
ax_j.set_title("J. Stochasticity", fontweight="bold")
ax_j.grid(True, alpha=0.3)
ax_j.legend(frameon=False, fontsize=6)

fig.subplots_adjust(hspace=0.5, wspace=0.35)
for out_name in ("Figure3Series_big_figure", "Figure3Series_big_editable"):
    fig.savefig(FIG_DIR / f"{out_name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{out_name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{out_name}.svg", bbox_inches="tight")
    print(f"Saved {out_name}.png/.pdf/.svg")
plt.close(fig)
print("✓ 已跳过 notebook 后续 scVelo velocity analysis / pancreas 统计验证。")
sys.exit(0)

# ===== Original notebook cell 18 =====

# 使用scvelo进行RNA Velocity分析
import scvelo as scv
import scanpy as sc

# 设置scvelo参数
scv.settings.verbosity = 3  # 显示详细信息
scv.settings.presenter_view = True  # 设置绘图视图
scv.set_figure_params('scvelo', dpi=100, facecolor='white', frameon=False)

print("=" * 60)
print("Step 1: 加载数据")
print("=" * 60)

# 如果adata已经存在，直接使用；否则从文件加载
if 'adata' not in locals() or adata is None:
    adata = sc.read_h5ad('scvelo_data.h5ad')
    print(f"✓ 从文件加载数据: {adata.n_obs} 个细胞, {adata.n_vars} 个基因")
else:
    print(f"✓ 使用已有数据: {adata.n_obs} 个细胞, {adata.n_vars} 个基因")

# 检查数据格式
print(f"\n数据检查:")
print(f"  - Layers: {list(adata.layers.keys())}")
print(f"  - obs列: {list(adata.obs.columns)}")
print(f"  - var列: {list(adata.var.columns)}")
print(f"  - obsm键: {list(adata.obsm.keys())}")

print("\n" + "=" * 60)
print("Step 2: 数据预处理和归一化")
print("=" * 60)

# 过滤和归一化（scvelo标准流程）
# 注意：如果数据已经是归一化的，可以跳过这一步
# 但scvelo需要特定的数据格式，所以建议执行

# 检查数据是否已经归一化
if 'spliced' in adata.layers and 'unspliced' in adata.layers:
    print("✓ 检测到spliced和unspliced layers")
    
    # 计算基本统计信息
    print(f"\n数据统计:")
    print(f"  Spliced - 均值: {adata.layers['spliced'].mean():.2f}, 最大值: {adata.layers['spliced'].max():.2f}")
    print(f"  Unspliced - 均值: {adata.layers['unspliced'].mean():.2f}, 最大值: {adata.layers['unspliced'].max():.2f}")
    
    # 在原始counts数据上加极微量噪声（防止重复点问题）
    # 这不会影响分析结果，因为噪声极小，且会被PCA降维过程自然过滤
    print("\n为原始counts数据添加极微量噪声（防止重复点，不影响流形结构）...")
    np.random.seed(42)
    
    # 计算噪声尺度（数据标准差的1e-6）
    spliced_std = np.std(adata.layers['spliced'])
    unspliced_std = np.std(adata.layers['unspliced'])
    noise_scale_s = spliced_std * 1e-6 if spliced_std > 0 else 1e-8
    noise_scale_u = unspliced_std * 1e-6 if unspliced_std > 0 else 1e-8
    
    # 添加极微量噪声
    noise_s = np.random.normal(0, noise_scale_s, size=adata.layers['spliced'].shape)
    noise_u = np.random.normal(0, noise_scale_u, size=adata.layers['unspliced'].shape)
    
    # 确保非负
    adata.layers['spliced'] = np.maximum(adata.layers['spliced'] + noise_s, 0)
    adata.layers['unspliced'] = np.maximum(adata.layers['unspliced'] + noise_u, 0)
    
    print(f"  ✓ 已添加极微量噪声（spliced: {noise_scale_s:.2e}, unspliced: {noise_scale_u:.2e}）")
    
    # 如果数据值很大（可能是原始计数），需要归一化
    if adata.layers['spliced'].max() > 100:
        print("\n执行过滤和归一化...")
        scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
        print("✓ 过滤和归一化完成")
    else:
        print("✓ 数据似乎已经归一化，跳过过滤步骤")
else:
    print("警告: 未找到spliced/unspliced layers，执行标准预处理...")
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)

print("\n" + "=" * 60)
print("Step 3: 计算Moments（用于velocity计算）")
print("=" * 60)

# 计算moments（需要降维坐标）
# 如果没有UMAP/PCA坐标，先计算
if 'X_umap' not in adata.obsm and 'X_pca' not in adata.obsm:
    print("未找到降维坐标，计算PCA和UMAP...")
    
    # 准备PCA数据
    # 如果数据值很大（原始计数），需要归一化
    if 'spliced' in adata.layers:
        max_val = adata.layers['spliced'].max()
        if max_val > 100:
            print("数据需要归一化，先执行归一化...")
            # 创建临时layer用于PCA
            adata.layers['spliced_normalized'] = adata.layers['spliced'].copy()
            sc.pp.normalize_total(adata, layer='spliced_normalized', target_sum=1e4)
            sc.pp.log1p(adata, layer='spliced_normalized')
            # 使用归一化后的数据作为X进行PCA
            adata.X = adata.layers['spliced_normalized'].copy()
        else:
            # 数据已经归一化，直接使用spliced作为X
            print("数据已归一化，使用spliced layer...")
            adata.X = adata.layers['spliced'].copy()
    
    # 计算PCA（不使用highly_variable，直接使用所有基因或前2000个）
    print("计算PCA...")
    # 如果基因太多，可以选择前N个方差最大的基因
    if adata.n_vars > 2000:
        print(f"  基因数较多({adata.n_vars})，选择前2000个方差最大的基因进行PCA...")
        gene_var = np.var(adata.X, axis=0)
        top_genes_idx = np.argsort(gene_var)[-2000:]
        adata_pca = adata[:, top_genes_idx].copy()
        sc.tl.pca(adata_pca, n_comps=30)
        # 将PCA结果复制回原始adata
        adata.obsm['X_pca'] = adata_pca.obsm['X_pca'].copy()
    else:
        sc.tl.pca(adata, n_comps=30)
    sc.pp.neighbors(adata, n_neighbors=30, n_pcs=30)
    sc.tl.umap(adata)
    print("✓ PCA和UMAP计算完成")
    
    # 也可以使用空间坐标
    if 'X_spatial' in adata.obsm:
        print("✓ 空间坐标可用，将用于可视化")
        # 保留X_spatial用于后续可视化

# 计算moments
print("计算moments...")
# 重新计算neighbors图以确保格式正确（避免警告）
print("  重新计算neighbors图（确保格式正确）...")
if 'neighbors' in adata.uns:
    del adata.uns['neighbors']
if 'distances' in adata.obsp:
    del adata.obsp['distances']
if 'connectivities' in adata.obsp:
    del adata.obsp['connectivities']

# 检查是否有PCA，如果没有则使用所有基因
if 'X_pca' in adata.obsm:
    print("  使用PCA坐标计算neighbors和moments...")
    sc.pp.neighbors(adata, n_neighbors=30, n_pcs=30)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
else:
    # 如果没有PCA，先计算neighbors图
    print("  没有PCA坐标，先计算neighbors图...")
    sc.pp.neighbors(adata, n_neighbors=30)
    scv.pp.moments(adata, n_neighbors=30)
print("✓ Moments计算完成")

print("\n" + "=" * 60)
print("Step 4: 计算RNA Velocity")
print("=" * 60)

# 对于dynamical模式，需要先运行recover_dynamics
print("恢复动力学参数（recover_dynamics）...")
try:
    scv.tl.recover_dynamics(adata)
    print("✓ 动力学参数恢复完成")
    
    # 计算velocity（使用dynamical模式）
    print("计算RNA velocity（dynamical模式）...")
    scv.tl.velocity(adata, mode='dynamical')
    print("✓ Velocity计算完成（dynamical模式）")
except Exception as e:
    print(f"警告: recover_dynamics失败 ({e})，使用stochastic模式...")
    scv.tl.velocity(adata, mode='stochastic')
    print("✓ Velocity计算完成（stochastic模式）")

# 重新计算neighbors图（为velocity_graph准备）
print("重新计算neighbors图...")
# 完全删除旧的neighbors图，确保格式正确
if 'neighbors' in adata.uns:
    del adata.uns['neighbors']
if 'distances' in adata.obsp:
    del adata.obsp['distances']
if 'connectivities' in adata.obsp:
    del adata.obsp['connectivities']

# 重新计算neighbors图
if 'X_pca' in adata.obsm:
    n_pcs = min(30, adata.obsm['X_pca'].shape[1])
    sc.pp.neighbors(adata, n_neighbors=50, n_pcs=n_pcs)
else:
    sc.pp.neighbors(adata, n_neighbors=50)

# 验证neighbors图是否正确
if 'distances' in adata.obsp and 'connectivities' in adata.obsp:
    n_neighs = (adata.obsp['distances'] > 0).sum(1).A1
    if np.min(n_neighs) == 0:
        print("警告: 某些细胞没有邻居，重新计算...")
        if 'X_pca' in adata.obsm:
            n_pcs = min(30, adata.obsm['X_pca'].shape[1])
            sc.pp.neighbors(adata, n_neighbors=50, n_pcs=n_pcs)
        else:
            sc.pp.neighbors(adata, n_neighbors=50)
        n_neighs = (adata.obsp['distances'] > 0).sum(1).A1
    print(f"✓ Neighbors图计算完成（每个细胞至少有 {np.min(n_neighs)} 个邻居）")
else:
    print("错误: Neighbors图格式不正确")
    raise ValueError("Neighbors图计算失败，请检查数据")

# 计算velocity graph
print("计算velocity graph...")
scv.tl.velocity_graph(adata)
print("✓ Velocity graph计算完成")

# 计算velocity pseudotime
print("计算velocity pseudotime...")
scv.tl.velocity_pseudotime(adata)
print("✓ Velocity pseudotime计算完成")

print("\n" + "=" * 60)
print("Step 5: 可视化结果")
print("=" * 60)

# 确保有PCA和UMAP坐标
if 'X_pca' not in adata.obsm:
    print("计算PCA...")
    sc.tl.pca(adata, n_comps=30)
    print("✓ PCA计算完成")

if 'X_umap' not in adata.obsm:
    print("计算UMAP...")
    if 'neighbors' not in adata.uns:
        sc.pp.neighbors(adata, n_neighbors=30, n_pcs=30)
    sc.tl.umap(adata)
    print("✓ UMAP计算完成")

# 计算velocity embedding（在PCA和UMAP空间中）
print("计算velocity embedding...")
scv.tl.velocity_embedding(adata, basis='pca')
scv.tl.velocity_embedding(adata, basis='umap')
if 'X_spatial' in adata.obsm:
    scv.tl.velocity_embedding(adata, basis='spatial')
print("✓ Velocity embedding计算完成")

# 创建可视化（美化版）
print("\n创建可视化...")

# ========== 第一组：Velocity Embedding ==========
n_rows = 3 if 'X_spatial' in adata.obsm else 2
fig1, axes1 = plt.subplots(n_rows, 3, figsize=(18, 6 * n_rows))
if n_rows == 2:
    axes1 = axes1.reshape(2, 3)
else:
    axes1 = axes1.reshape(3, 3)

row_idx = 0

# ========== PCA行 ==========
scv.pl.velocity_embedding(
    adata, basis='pca', ax=axes1[row_idx, 0], 
    show=False, title='Velocity Vectors (PCA)', 
    arrow_size=2, density=0.5, 
    frameon=False, legend_loc='none'
)

scv.pl.velocity_embedding_stream(
    adata, basis='pca', ax=axes1[row_idx, 1], 
    show=False, title='Velocity Stream (PCA)',
    frameon=False, legend_loc='none'
)

if 'time_point' in adata.obs:
    sc.pl.embedding(
        adata, basis='pca', color='time_point', 
        ax=axes1[row_idx, 2], show=False, 
        title='Time Points (PCA)', frameon=False
    )
else:
    sc.pl.pca(adata, ax=axes1[row_idx, 2], show=False, title='PCA', frameon=False)

row_idx += 1

# ========== UMAP行 ==========
scv.pl.velocity_embedding(
    adata, basis='umap', ax=axes1[row_idx, 0], 
    show=False, title='Velocity Vectors (UMAP)',
    arrow_size=2, density=0.5,
    frameon=False, legend_loc='none'
)

scv.pl.velocity_embedding_stream(
    adata, basis='umap', ax=axes1[row_idx, 1], 
    show=False, title='Velocity Stream (UMAP)',
    frameon=False, legend_loc='none'
)

if 'time_point' in adata.obs:
    sc.pl.umap(
        adata, color='time_point', 
        ax=axes1[row_idx, 2], show=False, 
        title='Time Points (UMAP)', frameon=False
    )
else:
    sc.pl.umap(adata, ax=axes1[row_idx, 2], show=False, title='UMAP', frameon=False)

row_idx += 1

# ========== Spatial行（如果可用）==========
if 'X_spatial' in adata.obsm:
    scv.pl.velocity_embedding(
        adata, basis='spatial', ax=axes1[row_idx, 0], 
        show=False, title='Velocity Vectors (Spatial)',
        arrow_size=2, density=0.5,
        frameon=False, legend_loc='none'
    )
    
    scv.pl.velocity_embedding_stream(
        adata, basis='spatial', ax=axes1[row_idx, 1], 
        show=False, title='Velocity Stream (Spatial)',
        frameon=False, legend_loc='none'
    )
    
    if 'time_point' in adata.obs:
        sc.pl.embedding(
            adata, basis='spatial', color='time_point', 
            ax=axes1[row_idx, 2], show=False, 
            title='Time Points (Spatial)', frameon=False
        )
    else:
        sc.pl.embedding(adata, basis='spatial', ax=axes1[row_idx, 2], show=False, title='Spatial', frameon=False)

# 美化：添加总标题
fig1.suptitle('RNA Velocity Analysis: Velocity Embeddings', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
save_current_figure('Figure3Series_velocity_metrics')

# ========== 第二组：Velocity Pseudotime和其他指标 ==========
print("\n创建第二组可视化...")
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))
axes2 = axes2.flatten()

# 1. Velocity pseudotime (PCA)
if 'velocity_pseudotime' in adata.obs:
    sc.pl.embedding(
        adata, basis='pca', color='velocity_pseudotime', 
        ax=axes2[0], show=False, 
        title='Velocity Pseudotime (PCA)', 
        cmap='viridis', frameon=False
    )

# 2. Velocity pseudotime (UMAP)
if 'velocity_pseudotime' in adata.obs:
    sc.pl.umap(
        adata, color='velocity_pseudotime', 
        ax=axes2[1], show=False, 
        title='Velocity Pseudotime (UMAP)', 
        cmap='viridis', frameon=False
    )

# 3. Velocity pseudotime (Spatial, 如果可用)
if 'velocity_pseudotime' in adata.obs and 'X_spatial' in adata.obsm:
    sc.pl.embedding(
        adata, basis='spatial', color='velocity_pseudotime', 
        ax=axes2[2], show=False, 
        title='Velocity Pseudotime (Spatial)', 
        cmap='viridis', frameon=False
    )
else:
    # 如果没有spatial，显示velocity长度
    adata.obs['velocity_length'] = np.linalg.norm(adata.layers['velocity'], axis=1)
    sc.pl.umap(
        adata, color='velocity_length', 
        ax=axes2[2], show=False, 
        title='Velocity Length (UMAP)', 
        cmap='plasma', frameon=False
    )

# 4. Velocity长度（PCA）
adata.obs['velocity_length'] = np.linalg.norm(adata.layers['velocity'], axis=1)
sc.pl.embedding(
    adata, basis='pca', color='velocity_length', 
    ax=axes2[3], show=False, 
    title='Velocity Length (PCA)', 
    cmap='plasma', frameon=False
)

# 5. Velocity长度（UMAP）
sc.pl.umap(
    adata, color='velocity_length', 
    ax=axes2[4], show=False, 
    title='Velocity Length (UMAP)', 
    cmap='plasma', frameon=False
)

# 6. 基因表达（选择一个cycle基因）
if 'cycle' in ' '.join(adata.var_names).lower():
    cycle_gene = [g for g in adata.var_names if 'cycle' in g.lower()][0]
    sc.pl.umap(
        adata, color=cycle_gene, 
        ax=axes2[5], show=False, 
        title=f'{cycle_gene} Expression (UMAP)', 
        cmap='Reds', frameon=False
    )
else:
    gene = adata.var_names[0]
    sc.pl.umap(
        adata, color=gene, 
        ax=axes2[5], show=False, 
        title=f'{gene} Expression (UMAP)', 
        cmap='Reds', frameon=False
    )

# 美化：添加总标题
fig2.suptitle('RNA Velocity Analysis: Pseudotime and Metrics', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
save_current_figure('Figure3Series_statistical_realism')

print("\n" + "=" * 60)
print("Step 6: 保存分析结果")
print("=" * 60)

# 保存分析后的数据
output_path_analyzed = "scvelo_analyzed.h5ad"
adata.write(output_path_analyzed)
print(f"✓ 分析结果已保存至: {output_path_analyzed}")

# 打印总结信息
print("\n" + "=" * 60)
print("分析总结")
print("=" * 60)
print(f"  - 总细胞数: {adata.n_obs}")
print(f"  - 总基因数: {adata.n_vars}")
print(f"  - Velocity模式: dynamical")
if 'velocity_pseudotime' in adata.obs:
    print(f"  - Pseudotime范围: [{adata.obs['velocity_pseudotime'].min():.2f}, {adata.obs['velocity_pseudotime'].max():.2f}]")
print(f"  - 数据已保存: {output_path_analyzed}")

print("\n✓ scvelo分析完成！")

# ===== Original notebook cell 19 =====

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scanpy as sc
import scvelo as scv
from scipy import sparse

print("正在绘制 Figure 2: Full Statistical Realism (6 Panels)...")

# ==========================================
# 1. 数据准备
# ==========================================

# --- A. 真实数据 (Pancreas Reference) ---

try:
    adata_real = scv.datasets.pancreas()
except Exception as e:
    print(f"[warn] scVelo pancreas dataset unavailable ({e}); using synthetic bootstrap reference.")
    adata_real = adata.copy()

# 确保有 spliced/unspliced
if 'spliced' not in adata_real.layers:
    # 如果下载的版本没有layers，模拟生成用于演示（实际使用请确保有）
    # scvelo内置pancreas通常是有的
    pass 

# --- B. 模拟数据 (Synthetic Target) ---
# 假设 adata 是你的模拟对象
# 确保 adata.layers['spliced'] 和 ['unspliced'] 存在
# 如果 X_sim 是 numpy array，请确保它是 (Cells, Genes) 格式

# 提取矩阵 (转为 dense array 以便计算)
def to_dense(X):
    return X.toarray() if sparse.issparse(X) else X

Xr_s = to_dense(adata_real.layers['spliced'])
Xr_u = to_dense(adata_real.layers['unspliced'])

Xs_s = to_dense(adata.layers['spliced'])
Xs_u = to_dense(adata.layers['unspliced'])

# 模拟数据如果维度不匹配 (Time x Gene)，展平处理
if Xs_s.ndim == 3: 
    Xs_s = Xs_s.reshape(-1, Xs_s.shape[-1])
    Xs_u = Xs_u.reshape(-1, Xs_u.shape[-1])

# ==========================================
# 2. 核心：计算 6 大统计指标
# ==========================================

def calc_6_metrics(S, U):
    """
    S: Spliced Matrix (Cells x Genes)
    U: Unspliced Matrix (Cells x Genes)
    """
    X_total = S + U
    
    # 1. Library Size (Per Cell)
    lib_size = X_total.sum(axis=1)
    
    # 2. Detected Genes (Per Cell) - Complexity
    n_genes = (X_total > 0).sum(axis=1)
    
    # 3. Mean Expression (Per Gene)
    gene_mean = X_total.mean(axis=0)
    
    # 4. Variance (Per Gene)
    gene_var = X_total.var(axis=0)
    
    # 5. Dropout Rate (Per Gene)
    # 过滤掉全0基因避免除零
    mask = gene_mean > 0
    mean_for_drop = gene_mean[mask]
    # 计算每个基因的0的数量 / 细胞总数
    drop_rate = (X_total[:, mask] == 0).sum(axis=0) / X_total.shape[0]
    
    # 6. Unspliced Fraction (Per Cell) - Critical for Velocity
    # u_sum / (s_sum + u_sum)
    u_sum = U.sum(axis=1)
    total_sum = lib_size
    # 避免除零
    total_sum[total_sum == 0] = 1 
    unspliced_frac = u_sum / total_sum
    
    return {
        'lib_size': lib_size,
        'n_genes': n_genes,
        'gene_mean': gene_mean,
        'gene_var': gene_var,
        'drop_mean': mean_for_drop, # 用于绘制 dropout 曲线的 x 轴
        'drop_rate': drop_rate,     # 用于绘制 dropout 曲线的 y 轴
        'u_frac': unspliced_frac
    }

print("计算统计指标...")
stats_real = calc_6_metrics(Xr_s, Xr_u)
stats_sim = calc_6_metrics(Xs_s, Xs_u)

# ==========================================
# 3. 绘图 (2x3 Grid)
# ==========================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=120)
plt.subplots_adjust(hspace=0.35, wspace=0.25)

# 风格设置
c_real = '#4A90E2' # 蓝 (Real)
c_sim = '#D0021B'  # 红 (Sim)
kwargs_kde = {'fill': True, 'common_norm': False, 'alpha': 0.3, 'linewidth': 2}

# --- Panel A: Library Size Distribution ---
ax = axes[0, 0]
sns.kdeplot(np.log1p(stats_real['lib_size']), color=c_real, ax=ax, label='Real (Pancreas)', **kwargs_kde)
sns.kdeplot(np.log1p(stats_sim['lib_size']), color=c_sim, ax=ax, label='Synthetic', **kwargs_kde)
ax.set_title('A. Library Size (Cell Depth)', fontweight='bold')
ax.set_xlabel('Log1p(Total Counts per Cell)')
ax.legend(frameon=False)
ax.grid(True, linestyle=':', alpha=0.3)

# --- Panel B: Detected Genes (Complexity) ---
ax = axes[0, 1]
sns.kdeplot(stats_real['n_genes'], color=c_real, ax=ax, **kwargs_kde)
sns.kdeplot(stats_sim['n_genes'], color=c_sim, ax=ax, **kwargs_kde)
ax.set_title('B. Detected Genes (Cell Complexity)', fontweight='bold')
ax.set_xlabel('Number of Detected Genes')
ax.grid(True, linestyle=':', alpha=0.3)

# --- Panel C: Gene Mean Distribution ---
ax = axes[0, 2]
# 过滤掉极低表达基因，避免图太难看
mask_r = stats_real['gene_mean'] > 0.01
mask_s = stats_sim['gene_mean'] > 0.01
sns.kdeplot(np.log1p(stats_real['gene_mean'][mask_r]), color=c_real, ax=ax, **kwargs_kde)
sns.kdeplot(np.log1p(stats_sim['gene_mean'][mask_s]), color=c_sim, ax=ax, **kwargs_kde)
ax.set_title('C. Mean Expression Distribution', fontweight='bold')
ax.set_xlabel('Log1p(Gene Mean Expression)')
ax.grid(True, linestyle=':', alpha=0.3)

# --- Panel D: Mean-Variance (Overdispersion) ---
ax = axes[1, 0]
# 随机采样 2000 个点避免过密
idx_r = np.random.choice(len(stats_real['gene_mean']), min(2000, len(stats_real['gene_mean'])), replace=False)
idx_s = np.random.choice(len(stats_sim['gene_mean']), min(2000, len(stats_sim['gene_mean'])), replace=False)

ax.scatter(np.log1p(stats_real['gene_mean'][idx_r]), np.log1p(stats_real['gene_var'][idx_r]), 
           s=10, c=c_real, alpha=0.3, label='Real')
ax.scatter(np.log1p(stats_sim['gene_mean'][idx_s]), np.log1p(stats_sim['gene_var'][idx_s]), 
           s=10, c=c_sim, alpha=0.3, label='Synthetic')

# Poisson Line
max_val = max(np.log1p(stats_real['gene_mean']).max(), np.log1p(stats_sim['gene_mean']).max())
line = np.linspace(0, max_val, 100)
ax.plot(line, line, 'k--', lw=1.5, label='Poisson')
ax.set_title('D. Mean-Variance (Overdispersion)', fontweight='bold')
ax.set_xlabel('Log1p(Mean)')
ax.set_ylabel('Log1p(Variance)')
ax.legend(frameon=False)
ax.grid(True, linestyle=':', alpha=0.3)

# --- Panel E: Dropout Rate ---
ax = axes[1, 1]
# 同样采样
idx_r_drop = np.random.choice(len(stats_real['drop_mean']), min(2000, len(stats_real['drop_mean'])), replace=False)
idx_s_drop = np.random.choice(len(stats_sim['drop_mean']), min(2000, len(stats_sim['drop_mean'])), replace=False)

ax.scatter(np.log1p(stats_real['drop_mean'][idx_r_drop]), stats_real['drop_rate'][idx_r_drop], 
           s=10, c=c_real, alpha=0.3)
ax.scatter(np.log1p(stats_sim['drop_mean'][idx_s_drop]), stats_sim['drop_rate'][idx_s_drop], 
           s=10, c=c_sim, alpha=0.3)
ax.set_title('E. Dropout Rate vs Mean', fontweight='bold')
ax.set_xlabel('Log1p(Mean)')
ax.set_ylabel('Fraction of Zeros')
ax.grid(True, linestyle=':', alpha=0.3)

# --- Panel F: Unspliced Fraction (Velocity Specific) ---
ax = axes[1, 2]
sns.kdeplot(stats_real['u_frac'], color=c_real, ax=ax, **kwargs_kde)
sns.kdeplot(stats_sim['u_frac'], color=c_sim, ax=ax, **kwargs_kde)
ax.set_title('F. Unspliced Fraction (Global Kinetics)', fontweight='bold')
ax.set_xlabel('Unspliced Reads / Total Reads')
ax.set_xlim(0, 0.5) # 通常u的比例不会超过50%
ax.grid(True, linestyle=':', alpha=0.3)

plt.tight_layout()
save_current_figure()

print("✓ 6-Panel Statistical Validation Plot Generated.")