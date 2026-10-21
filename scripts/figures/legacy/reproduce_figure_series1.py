"""
Legacy notebook-export script. Prefer the panelized packages under scripts/figures/ (see scripts/figures/README.md).
"""


# ===== Original notebook cell 2 =====

# 导入必要的库
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
from pathlib import Path
import sys

# 添加项目根目录到路径
# 获取项目根目录（notebook所在目录的父目录）
project_root = Path.cwd().parent if Path.cwd().name == 'figure_notebook' else Path.cwd()
sys.path.insert(0, str(project_root))
print(f"项目根目录: {project_root}")

# 尝试导入数据生成模块
try:
    from data_generation_module import GraphNode, CycleNode, GraphEdge, UniversalGraphSystem
    from data_generation_module import GraphSimulator
    from data_generation_module import GeneSynthesizer
    from data import generate_synthetic_data
except ImportError as e:
    print(f"警告: 无法导入数据生成模块: {e}")
    print("将使用模拟数据")

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

OUT_DIR = project_root / 'data' / 'results' / 'figure_series1'
FIG_DIR = OUT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
print(f'图片输出目录: {FIG_DIR}')

# 定义配色方案
color_palette = {
    'deep_blue_gray': '#396185',    # 深蓝灰色
    'blue_gray': '#6981a3',         # 蓝灰色
    'light_blue': '#93b1d1',        # 浅蓝色
    'lighter_blue': '#a8c9e8',      # 更浅的蓝色
    'very_light_blue': '#bde0fe',   # 非常浅的蓝色
    'light_purple': '#ded4ee',      # 浅紫色
    'light_pink': '#ffc8dd',        # 浅粉色
    'pink': '#f1b5c8',              # 粉色
    'deep_pink': '#e2a2b3',         # 深粉色
    'deeper_pink': '#db8c9f',       # 更深粉色
    'deepest_pink': '#cb7892'       # 最深粉色
}

print("✓ 库导入完成")
print(f"✓ 配色方案已加载: {len(color_palette)} 种颜色")

# ===== Original notebook cell 3 =====

# 生成Y形拓扑的完整轨迹数据
np.random.seed(42)
np.random.seed(42)

# 参数设置
N_samples = 2000  # 增加样本数以获得更好的UMAP
D_genes = 50
noise_level = 0.5
d_latent = 2
n_steps = 100  # 生成更多时间步用于完整轨迹
dt = 0.05

print("生成Y形拓扑数据...")

# 构建Y形拓扑: Stem -> Prog(Cycle) -> FateA/FateB
nodes = [
    GraphNode("Stem", [-2.0, 0], potential=0.0, sigma=0.2, basin_sigma=0.5, basin_strength=0.2),
    CycleNode("Prog", [0.0, 0], potential=-1.5, sigma=0.2, radius=1.0, omega=5.0, clockwise=False),
    GraphNode("FateA", [2.0, 1.5], potential=-4.0, sigma=0.15),
    GraphNode("FateB", [2.0, -1.5], potential=-4.0, sigma=0.15),
]

edges = [
    GraphEdge(nodes[0], nodes[1]),  # Edge 0: Stem -> Prog
    GraphEdge(nodes[1], nodes[2]),  # Edge 1: Prog -> FateA
    GraphEdge(nodes[1], nodes[3]),  # Edge 2: Prog -> FateB
]

system = UniversalGraphSystem(nodes, edges, fusion_temp=0.05)

# 生成完整轨迹
print(f"  模拟 {N_samples} 个细胞，{n_steps} 个时间步...")
traj = GraphSimulator.run(system, "Stem", n_cells=N_samples, steps=n_steps, dt=dt, noise_level=noise_level)
# traj.shape == (n_steps+1, N_samples, 2)

# 创建基因合成器
synthesizer_kwargs = {
    'edge_strength': 5.0,
    'cycle_strength': 3.0,
    'background_noise': 0.1
}
synth = GeneSynthesizer(system, n_genes_total=D_genes, **synthesizer_kwargs)

# 分配基因到不同模块
# 增加sticky基因的分配
# Edge0, Cycle, Edge1, Edge2, Background, Sticky
fracs = np.array([0.25, 0.2, 0.15, 0.15, 0.1, 0.15])  # 添加15%的sticky基因
counts = (fracs * D_genes).astype(int)
diff = D_genes - counts.sum()
for i in range(abs(diff)):
    counts[i % len(counts)] += int(np.sign(diff))

# 添加基因模块
n_e0, n_cycle, n_e1, n_e2, n_bg, n_sticky = counts
if n_e0 > 0:
    synth.add_trend_gene(edge_idx=0, n_genes=n_e0, pattern='sigmoid', reverse=False)
if n_cycle > 0:
    synth.add_cycle_gene(node_id='Prog', n_genes=n_cycle)
if n_e1 > 0:
    synth.add_trend_gene(edge_idx=1, n_genes=n_e1, pattern='sigmoid', reverse=False)
if n_e2 > 0:
    synth.add_trend_gene(edge_idx=2, n_genes=n_e2, pattern='sigmoid', reverse=False)
if n_bg > 0:
    synth.add_trend_gene(edge_idx=0, n_genes=n_bg, pattern='impulse', reverse=True)

# 添加sticky基因（分布在不同的edge上）
# Sticky基因在CycleNode附近会保持高表达，有助于velocity推断
if n_sticky > 0:
    # 将sticky基因分配到Edge1和Edge2（分叉后的edge）
    n_sticky_e1 = n_sticky // 2
    n_sticky_e2 = n_sticky - n_sticky_e1
    
    if n_sticky_e1 > 0:
        synth.add_trend_gene(edge_idx=1, n_genes=n_sticky_e1, pattern='sigmoid', reverse=False, sticky=True)
    if n_sticky_e2 > 0:
        synth.add_trend_gene(edge_idx=2, n_genes=n_sticky_e2, pattern='sigmoid', reverse=False, sticky=True)
    
    print(f"  添加了 {n_sticky} 个sticky基因 (Edge1: {n_sticky_e1}, Edge2: {n_sticky_e2})")

# 新的采样策略：从最后一个时间采样一半，从之前的时间均匀采样剩下的一半
print("  应用新的采样策略...")
n_time_steps = traj.shape[0]  # n_steps + 1
n_cells_per_step = traj.shape[1]  # N_samples

# 计算需要采样的细胞数
n_cells_to_sample = N_samples*2
n_from_last = n_cells_to_sample // 2  # 从最后一个时间点采样一半
n_from_earlier = n_cells_to_sample - n_from_last  # 从之前的时间点采样剩下的一半

# 从最后一个时间点采样
np.random.seed(42)
last_time_indices = np.random.choice(n_cells_per_step, size=n_from_last, replace=False)
z_last = traj[-1, last_time_indices, :]  # [n_from_last, 2]

# 从之前的时间点均匀采样（排除最后一个时间点）
earlier_time_steps = np.arange(0, n_time_steps - 1)  # 排除最后一个
n_steps_to_sample = len(earlier_time_steps)
cells_per_earlier_step = n_from_earlier // n_steps_to_sample
remaining = n_from_earlier % n_steps_to_sample

z_earlier_list = []
for i, t_idx in enumerate(earlier_time_steps):
    n_cells_this_step = cells_per_earlier_step + (1 if i < remaining else 0)
    if n_cells_this_step > 0:
        step_indices = np.random.choice(n_cells_per_step, size=min(n_cells_this_step, n_cells_per_step), replace=False)
        z_earlier_list.append(traj[t_idx, step_indices, :])

if z_earlier_list:
    z_earlier = np.concatenate(z_earlier_list, axis=0)  # [n_from_earlier, 2]
else:
    z_earlier = np.empty((0, 2))

# 合并采样结果
z_final = np.concatenate([z_last, z_earlier], axis=0)  # [n_cells_to_sample, 2]
z_final = np.asarray(z_final, dtype=np.float32)

# 记录每个细胞来自哪个时间点（用于后续分析）
time_indices = np.concatenate([
    np.full(n_from_last, n_time_steps - 1),  # 最后一个时间点
    np.repeat(earlier_time_steps, [len(z) for z in z_earlier_list] if z_earlier_list else [0])
])

print(f"  采样完成: 从最后时间点={n_from_last}, 从之前时间点={n_from_earlier}, 总计={len(z_final)}")

# 保存采样信息（用于后续提取spliced/unspliced数据）
# 构建完整轨迹中的采样索引
sampled_indices_full_traj = []

# 从最后一个时间点的采样
for idx in last_time_indices:
    time_idx = n_time_steps - 1
    cell_idx_in_full = time_idx * n_cells_per_step + idx
    sampled_indices_full_traj.append(cell_idx_in_full)

# 从之前时间点的采样（需要重新执行以确保一致性）
np.random.seed(42)  # 重置随机种子
for i, t_idx in enumerate(earlier_time_steps):
    n_cells_this_step = cells_per_earlier_step + (1 if i < remaining else 0)
    if n_cells_this_step > 0:
        step_indices = np.random.choice(n_cells_per_step, 
                                       size=min(n_cells_this_step, n_cells_per_step), 
                                       replace=False)
        for idx in step_indices:
            cell_idx_in_full = t_idx * n_cells_per_step + idx
            sampled_indices_full_traj.append(cell_idx_in_full)

sampled_indices_full_traj = np.array(sampled_indices_full_traj)

# 生成表达数据（使用新的采样策略，用于UMAP等可视化）
X_final = synth.generate(z_final, simulate_counts=False)  # 连续表达率

print(f"✓ 数据生成完成")
print(f"  轨迹形状: {traj.shape}")
print(f"  采样后坐标形状: {z_final.shape}")
print(f"  表达数据形状: {X_final.shape}")
print(f"  采样索引形状: {sampled_indices_full_traj.shape}")
print(f"  基因模块分布: Edge0={n_e0}, Cycle={n_cycle}, Edge1={n_e1}, Edge2={n_e2}, Background={n_bg}, Sticky={n_sticky}")


# ===== Original notebook cell 5 =====

# 计算pseudotime和branch信息
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import umap
import scanpy as sc
import scvelo as scv
import anndata as ad
from scipy import sparse

print("计算pseudotime和branch信息...")

# 转换z_final为numpy（已经是 numpy）
z_final_np = z_final if isinstance(z_final, np.ndarray) else z_final
n_cells_sampled = z_final_np.shape[0]

# 使用潜在空间坐标计算pseudotime
# pseudotime定义为从Stem节点到当前细胞的距离
stem_pos = np.array([[-2.0, 0.0]])  # Stem节点位置
distances_to_stem = cdist(z_final_np, stem_pos).flatten()

# 归一化pseudotime到[0, 1]
pseudotime = (distances_to_stem - distances_to_stem.min()) / (distances_to_stem.max() - distances_to_stem.min() + 1e-10)

# 计算branch信息：根据细胞在潜在空间中的位置判断属于哪个分支
branch = np.zeros(n_cells_sampled, dtype=int)  # 0: Stem/Prog, 1: FateA, 2: FateB

# 定义分支判断阈值
fateA_pos = np.array([[2.0, 1.5]])
fateB_pos = np.array([[2.0, -1.5]])

dist_to_fateA = cdist(z_final_np, fateA_pos).flatten()
dist_to_fateB = cdist(z_final_np, fateB_pos).flatten()

# 如果距离FateA更近且x坐标 > 0.5，则为FateA分支
# 如果距离FateB更近且x坐标 > 0.5，则为FateB分支
branch[dist_to_fateA < dist_to_fateB] = 1  # FateA
branch[dist_to_fateB < dist_to_fateA] = 2  # FateB
# 默认0表示Stem/Prog

# 计算UMAP降维
print("  计算UMAP...")
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
z_umap = reducer.fit_transform(z_final_np)

print(f"✓ Pseudotime和Branch计算完成")
print(f"  Pseudotime范围: [{pseudotime.min():.3f}, {pseudotime.max():.3f}]")
print(f"  Branch分布: Stem/Prog={np.sum(branch==0)}, FateA={np.sum(branch==1)}, FateB={np.sum(branch==2)}")

# 生成spliced/unspliced数据用于velocity计算
print("  生成spliced/unspliced数据用于velocity计算...")
# 关键：使用完整轨迹生成数据，这样才能有正确的动力学方向
# 将完整轨迹转为 numpy 数组
traj_np = np.asarray(traj, dtype=np.float32)  # [n_steps+1, N_samples, 2]

# 使用完整轨迹生成真实的spliced/unspliced数据
print("    使用完整轨迹生成真实的spliced/unspliced（包含正确的动力学方向）...")
kinetics_result_full = synth.generate_kinetics(
    traj_np, 
    dt=dt, 
    simulate_counts=True,
    scale=1.0
)

# 提取完整轨迹的spliced/unspliced
Xs_s_full = kinetics_result_full['spliced']
Xs_u_full = kinetics_result_full['unspliced']

# 转换为numpy
if isinstance(Xs_s_full, np.ndarray):
    Xs_s_full = Xs_s_full
if isinstance(Xs_u_full, np.ndarray):
    Xs_u_full = Xs_u_full

# 如果是3D，展平为2D [n_steps+1 * N_samples, n_genes]
if Xs_s_full.ndim == 3:
    Xs_s_full = Xs_s_full.reshape(-1, Xs_s_full.shape[-1])
    Xs_u_full = Xs_u_full.reshape(-1, Xs_u_full.shape[-1])

# 现在需要根据采样策略提取对应的数据
# 使用Cell 3中保存的采样索引
print("    根据采样策略提取对应的spliced/unspliced数据...")

# 检查采样索引是否已定义（从Cell 3）
if 'sampled_indices_full_traj' not in globals():
    # 如果没有，重新构建（使用相同的随机种子）
    print("    重建采样索引...")
    np.random.seed(42)
    sampled_indices_full_traj = []
    
    # 从最后一个时间点的采样
    last_time_indices_rebuild = np.random.choice(n_cells_per_step, size=n_from_last, replace=False)
    for idx in last_time_indices_rebuild:
        time_idx = n_time_steps - 1
        cell_idx_in_full = time_idx * n_cells_per_step + idx
        sampled_indices_full_traj.append(cell_idx_in_full)
    
    # 从之前时间点的采样
    for i, t_idx in enumerate(earlier_time_steps):
        n_cells_this_step = cells_per_earlier_step + (1 if i < remaining else 0)
        if n_cells_this_step > 0:
            step_indices = np.random.choice(n_cells_per_step, 
                                           size=min(n_cells_this_step, n_cells_per_step), 
                                           replace=False)
            for idx in step_indices:
                cell_idx_in_full = t_idx * n_cells_per_step + idx
                sampled_indices_full_traj.append(cell_idx_in_full)
    
    sampled_indices_full_traj = np.array(sampled_indices_full_traj)

# 确保索引在有效范围内
sampled_indices_full_traj = sampled_indices_full_traj[sampled_indices_full_traj < Xs_s_full.shape[0]]

# 提取采样细胞的数据
Xs_s = Xs_s_full[sampled_indices_full_traj]
Xs_u = Xs_u_full[sampled_indices_full_traj]

# 确保数据形状匹配
if len(Xs_s) != n_cells_sampled:
    print(f"    警告: 采样数据形状不匹配 ({len(Xs_s)} vs {n_cells_sampled})，调整中...")
    if len(Xs_s) > n_cells_sampled:
        Xs_s = Xs_s[:n_cells_sampled]
        Xs_u = Xs_u[:n_cells_sampled]
    else:
        # 如果不够，从完整数据中补充
        n_needed = n_cells_sampled - len(Xs_s)
        available_indices = np.setdiff1d(np.arange(Xs_s_full.shape[0]), sampled_indices_full_traj)
        if len(available_indices) >= n_needed:
            additional_indices = np.random.choice(available_indices, size=n_needed, replace=False)
            Xs_s = np.concatenate([Xs_s, Xs_s_full[additional_indices]], axis=0)
            Xs_u = np.concatenate([Xs_u, Xs_u_full[additional_indices]], axis=0)
        else:
            # 如果可用索引不够，使用重复
            Xs_s = np.concatenate([Xs_s, Xs_s_full[sampled_indices_full_traj[-n_needed:]]], axis=0)
            Xs_u = np.concatenate([Xs_u, Xs_u_full[sampled_indices_full_traj[-n_needed:]]], axis=0)

print(f"    完整轨迹数据形状: {Xs_s_full.shape}")
print(f"    采样后数据形状: {Xs_s.shape}")

# 创建AnnData对象
adata = ad.AnnData(X=Xs_s)
adata.layers['spliced'] = Xs_s
adata.layers['unspliced'] = Xs_u

# 添加元数据
adata.obs['pseudotime'] = pseudotime
adata.obs['branch'] = branch.astype(str)
adata.obs['time_index'] = time_indices
adata.obsm['X_umap'] = z_umap
adata.obsm['X_trajectory'] = z_final_np

# 预处理
print("  预处理数据...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# 动态确定PCA组件数（必须小于min(n_samples, n_features)）
n_top_genes = min(30, adata.n_vars)
sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=False)

# PCA组件数必须严格小于min(n_samples, n_features)（arpack要求）
min_dim = min(adata.n_obs, adata.n_vars)
print(f"    数据形状: {adata.shape}")
print(f"    min(n_samples, n_features) = {min_dim}")

# 计算n_comps：严格小于min_dim
if min_dim <= 1:
    n_comps = 1
elif min_dim <= 30:
    n_comps = min_dim - 1  # 严格小于min_dim
else:
    n_comps = 30  # 当min_dim > 30时，可以使用30

# 确保n_comps至少为1
n_comps = max(1, n_comps)

# 双重检查：确保n_comps严格小于min_dim
if n_comps >= min_dim:
    n_comps = max(1, min_dim - 1)
    print(f"    警告: 调整n_comps以确保严格小于min_dim")

print(f"    高变基因数: {n_top_genes}")
print(f"    PCA组件数: {n_comps} (必须 < {min_dim})")

# 根据数据大小选择合适的svd_solver
# 使用'auto'让sklearn自动选择最佳方法，避免arpack的严格限制
svd_solver = 'auto'

sc.tl.pca(adata, n_comps=n_comps, svd_solver=svd_solver)
sc.pp.neighbors(adata, n_neighbors=min(15, adata.n_obs-1), n_pcs=n_comps)

# 计算velocity
print("  计算RNA velocity...")
n_neighbors_vel = min(30, adata.n_obs - 1)
scv.pp.moments(adata, n_neighbors=n_neighbors_vel, n_pcs=n_comps)
scv.tl.velocity(adata, mode='stochastic')
scv.tl.velocity_graph(adata)
scv.tl.velocity_embedding(adata, basis='umap')

# 检查并修正velocity方向
# 注意：对于CycleNode（环结构），velocity可能沿着环移动，这是正常的
# 我们只检查明显的方向错误（velocity指向Stem而不是离开Stem）
print("  检查velocity方向...")
if 'velocity_umap' in adata.obsm and 'X_trajectory' in adata.obsm:
    velocity_umap = adata.obsm['velocity_umap']
    trajectory_coords = adata.obsm['X_trajectory']
    umap_coords = adata.obsm['X_umap']
    
    # 使用真实轨迹的时间方向作为ground truth
    # 从完整轨迹中提取每个采样细胞对应的真实速度方向
    print("    从完整轨迹计算真实速度方向...")
    
    # 重建采样细胞在完整轨迹中的时间索引
    # 使用time_indices来找到每个细胞在轨迹中的位置
    true_velocity_2d = np.zeros_like(trajectory_coords)
    
    # 对于每个采样细胞，找到它在完整轨迹中的位置，然后计算其真实速度
    for i, time_idx in enumerate(time_indices):
        if time_idx < n_time_steps - 1:
            # 找到这个细胞在完整轨迹中的索引
            # 需要找到对应的原始细胞索引
            # 简化：使用轨迹坐标的最近邻匹配
            
            # 找到当前时间点的所有细胞坐标
            traj_at_t = traj[time_idx]  # [N_samples, 2]
            traj_at_t_next = traj[time_idx + 1]  # [N_samples, 2]
            
            # 找到当前采样细胞在traj_at_t中的最近邻
            nbrs = NearestNeighbors(n_neighbors=1).fit(traj_at_t)
            distances, indices = nbrs.kneighbors(trajectory_coords[i:i+1])
            
            nearest_idx = indices[0, 0]
            
            # 计算真实速度（在2D空间中）
            true_vel_2d = traj_at_t_next[nearest_idx] - traj_at_t[nearest_idx]
            true_velocity_2d[i] = true_vel_2d
        else:
            # 最后一个时间点，使用前一个时间点的速度
            if time_idx > 0:
                traj_at_t = traj[time_idx - 1]
                traj_at_t_next = traj[time_idx]
                nbrs = NearestNeighbors(n_neighbors=1).fit(traj_at_t)
                distances, indices = nbrs.kneighbors(trajectory_coords[i:i+1])
                nearest_idx = indices[0, 0]
                true_vel_2d = traj_at_t_next[nearest_idx] - traj_at_t[nearest_idx]
                true_velocity_2d[i] = true_vel_2d
    
    # 将真实速度方向投影到UMAP空间
    # 使用PCA或最近邻映射
    
    # 方法：在UMAP空间中，找到与真实2D速度方向一致的UMAP方向
    # 使用k近邻计算局部映射
    nbrs = NearestNeighbors(n_neighbors=min(10, len(umap_coords)-1)).fit(umap_coords)
    distances, indices = nbrs.kneighbors(umap_coords)
    
    true_direction_umap = np.zeros_like(velocity_umap)
    
    for i in range(len(umap_coords)):
        neighbors = indices[i][1:]  # 排除自己
        if len(neighbors) > 0:
            # 计算真实2D速度在邻居中的投影
            true_vel_2d_i = true_velocity_2d[i]
            true_vel_norm_2d = np.linalg.norm(true_vel_2d_i)
            
            if true_vel_norm_2d > 1e-6:
                # 找到在2D空间中与真实速度方向一致的邻居
                traj_neighbors = trajectory_coords[neighbors]
                traj_diff = traj_neighbors - trajectory_coords[i]
                
                # 计算traj_diff与true_vel_2d_i的点积
                dot_products = (traj_diff * true_vel_2d_i).sum(axis=1)
                traj_norms = np.sqrt((traj_diff**2).sum(axis=1))
                traj_norms[traj_norms == 0] = 1.0
                cosines = dot_products / (true_vel_norm_2d * traj_norms)
                
                # 只考虑方向一致的邻居（cosine > 0.5）
                positive_mask = cosines > 0.5
                if positive_mask.sum() > 0:
                    umap_grad = umap_coords[neighbors[positive_mask]] - umap_coords[i]
                    # 归一化
                    grad_norm = np.sqrt((umap_grad**2).sum(axis=1, keepdims=True))
                    grad_norm[grad_norm == 0] = 1.0
                    umap_grad_normalized = umap_grad / grad_norm
                    # 加权平均（权重为cosine值）
                    weights = cosines[positive_mask]
                    true_direction_umap[i] = (umap_grad_normalized * weights[:, np.newaxis]).sum(axis=0) / weights.sum()
    
    # 计算velocity方向与真实方向的相关性
    velocity_direction_normalized = velocity_umap.copy()
    vel_norms = np.sqrt((velocity_umap**2).sum(axis=1, keepdims=True))
    vel_norms[vel_norms == 0] = 1.0
    velocity_direction_normalized = velocity_umap / vel_norms
    
    true_dir_norms = np.sqrt((true_direction_umap**2).sum(axis=1, keepdims=True))
    true_dir_norms[true_dir_norms == 0] = 1.0
    true_direction_normalized = true_direction_umap / true_dir_norms
    
    # 计算点积（相关性）
    correlations = (velocity_direction_normalized * true_direction_normalized).sum(axis=1)
    valid_mask = (vel_norms.flatten() > 0) & (true_dir_norms.flatten() > 0)
    
    if valid_mask.sum() > 0:
        mean_correlation = correlations[valid_mask].mean()
        print(f"    Velocity与真实轨迹方向的相关性: {mean_correlation:.3f}")
        
        # 更保守的检查：只对明显错误的情况进行修正
        # 对于CycleNode，velocity可能沿着环移动，这是正常的
        # 我们只检查是否明显指向错误的方向（比如指向Stem）
        
        # 检查：对于远离Stem的细胞，velocity是否指向Stem
        stem_pos_2d = np.array([-2.0, 0.0])
        dist_to_stem = np.sqrt(((trajectory_coords - stem_pos_2d)**2).sum(axis=1))
        
        # 对于距离Stem较远的细胞（>1.0），检查velocity是否指向Stem
        far_from_stem = dist_to_stem > 1.0
        if far_from_stem.sum() > 0:
            # 计算这些细胞的velocity是否指向Stem
            far_cells_vel = velocity_umap[far_from_stem]
            far_cells_pos = trajectory_coords[far_from_stem]
            far_cells_to_stem = stem_pos_2d - far_cells_pos
            
            # 归一化
            vel_norms_far = np.sqrt((far_cells_vel**2).sum(axis=1, keepdims=True))
            vel_norms_far[vel_norms_far == 0] = 1.0
            vel_normalized = far_cells_vel / vel_norms_far
            
            to_stem_norms = np.sqrt((far_cells_to_stem**2).sum(axis=1, keepdims=True))
            to_stem_norms[to_stem_norms == 0] = 1.0
            to_stem_normalized = far_cells_to_stem / to_stem_norms
            
            # 计算点积（如果>0.5，说明velocity指向Stem，这是错误的）
            pointing_to_stem = (vel_normalized * to_stem_normalized).sum(axis=1)
            wrong_direction_ratio = (pointing_to_stem > 0.5).mean()
            
            print(f"    远离Stem的细胞中，velocity指向Stem的比例: {wrong_direction_ratio:.3f}")
            
            # 如果大部分细胞都指向Stem，说明方向反了
            if wrong_direction_ratio > 0.6 and mean_correlation < -0.2:
                print(f"    警告: 大部分velocity指向Stem（错误方向），正在反转...")
                adata.layers['velocity'] = -adata.layers['velocity']
                if 'velocity_umap' in adata.obsm:
                    adata.obsm['velocity_umap'] = -adata.obsm['velocity_umap']
                # 重新计算velocity graph
                scv.tl.velocity_graph(adata)
                print(f"    ✓ Velocity方向已反转")
            elif mean_correlation > 0.1:
                print(f"    ✓ Velocity方向基本正确")
            else:
                print(f"    ⚠ Velocity方向检查：相关性={mean_correlation:.3f}，指向Stem比例={wrong_direction_ratio:.3f}")
                print(f"       (对于CycleNode，velocity沿环移动是正常的)")
        else:
            # 如果没有远离Stem的细胞，使用相关性判断
            if mean_correlation < -0.3:
                print(f"    警告: Velocity方向与真实轨迹方向明显相反，正在反转...")
                adata.layers['velocity'] = -adata.layers['velocity']
                if 'velocity_umap' in adata.obsm:
                    adata.obsm['velocity_umap'] = -adata.obsm['velocity_umap']
                scv.tl.velocity_graph(adata)
                print(f"    ✓ Velocity方向已反转")
            else:
                print(f"    ✓ Velocity方向检查完成（相关性={mean_correlation:.3f}）")

print(f"✓ Velocity计算完成")


# ===== Original notebook cell 9 =====

# 绘制Panel A: Topology示意图和UMAP
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左图: Y形拓扑示意图
ax_left = axes[0]
ax_left.set_xlim(-3, 3)
ax_left.set_ylim(-2.5, 2.5)
ax_left.set_aspect('equal')
ax_left.axis('off')

# 绘制节点
node_positions = {
    'Stem': (-2.0, 0.0),
    'Prog': (0.0, 0.0),
    'FateA': (2.0, 1.5),
    'FateB': (2.0, -1.5)
}

node_colors = {
    'Stem': color_palette['deep_blue_gray'],    # 深蓝灰色
    'Prog': color_palette['blue_gray'],        # 蓝灰色
    'FateA': color_palette['light_blue'],       # 浅蓝色
    'FateB': color_palette['light_pink']        # 浅粉色
}

# 绘制边（箭头）
edges_list = [
    ('Stem', 'Prog'),
    ('Prog', 'FateA'),
    ('Prog', 'FateB')
]

for start, end in edges_list:
    x1, y1 = node_positions[start]
    x2, y2 = node_positions[end]
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->', 
        lw=2.5,
        color=color_palette['deep_blue_gray'],
        zorder=1
    )
    ax_left.add_patch(arrow)

# 绘制节点（圆形）
for node_name, (x, y) in node_positions.items():
    circle = plt.Circle((x, y), 0.3, color=node_colors[node_name], 
                       ec='black', lw=2, zorder=2)
    ax_left.add_patch(circle)
    ax_left.text(x, y, node_name, ha='center', va='center', 
                fontsize=12, fontweight='bold', zorder=3)

ax_left.set_title('Input Graph (Y-shaped Topology)', fontsize=14, fontweight='bold', pad=20)

# 右图: UMAP可视化
ax_right = axes[1]

# 根据branch着色
branch_colors = [
    color_palette['blue_gray'],      # Stem/Prog - 蓝灰色
    color_palette['light_blue'],     # FateA - 浅蓝色
    color_palette['light_pink']       # FateB - 浅粉色
]
branch_labels = ['Stem/Prog', 'FateA', 'FateB']

for br in [0, 1, 2]:
    mask = branch == br
    ax_right.scatter(z_umap[mask, 0], z_umap[mask, 1], 
                    c=branch_colors[br], label=branch_labels[br], 
                    alpha=0.6, s=20, edgecolors='none')

ax_right.set_xlabel('UMAP 1', fontsize=12)
ax_right.set_ylabel('UMAP 2', fontsize=12)
ax_right.set_title('UMAP Visualization (colored by branch)', fontsize=14, fontweight='bold')
ax_right.legend(loc='best', fontsize=10)
ax_right.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / 'Fig1_PanelA.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Panel A绘制完成")


# ===== Original notebook cell 13 =====

# 补齐 notebook 中未显式定义的 phase portrait / time-series 变量。
# 选择平均表达随时间变化最大的基因，保证 u/s 轨迹有可见动态。
Xs_s_full_3d = Xs_s_full.reshape(n_time_steps, n_cells_per_step, -1)
Xs_u_full_3d = Xs_u_full.reshape(n_time_steps, n_cells_per_step, -1)
mean_s_by_time_gene = Xs_s_full_3d.mean(axis=1)
mean_u_by_time_gene = Xs_u_full_3d.mean(axis=1)
sampled_s_std = Xs_s.std(axis=0)
sampled_u_std = Xs_u.std(axis=0)
time_s_std = mean_s_by_time_gene.std(axis=0)
time_u_std = mean_u_by_time_gene.std(axis=0)
gene_dynamic_score = (sampled_s_std * sampled_u_std) + 0.1 * (time_s_std * time_u_std)
phase_gene_idx = int(np.argmax(gene_dynamic_score))

s_values = Xs_s[:, phase_gene_idx]
u_values = Xs_u[:, phase_gene_idx]
t_series = np.arange(n_time_steps) * dt
s_series = mean_s_by_time_gene[:, phase_gene_idx]
u_series = mean_u_by_time_gene[:, phase_gene_idx]
s_peak_idx = int(np.argmax(s_series))
u_peak_idx = int(np.argmax(u_series))
s_peak_time = float(t_series[s_peak_idx])
u_peak_time = float(t_series[u_peak_idx])
print(f"  Phase/time-series gene index: {phase_gene_idx}")

# 创建组合图：包含拓扑、UMAP、Phase Portrait和Velocity可视化
fig = plt.figure(figsize=(24, 12))

# Panel A (左) - 拓扑图
ax1 = plt.subplot(2, 3, 1)
ax1.set_xlim(-3, 3)
ax1.set_ylim(-2.5, 2.5)
ax1.set_aspect('equal')
ax1.axis('off')

# 绘制Y形拓扑
for start, end in edges_list:
    x1, y1 = node_positions[start]
    x2, y2 = node_positions[end]
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->', 
        lw=2.5,
        color=color_palette['deep_blue_gray'],
        zorder=1
    )
    ax1.add_patch(arrow)

for node_name, (x, y) in node_positions.items():
    circle = plt.Circle((x, y), 0.3, color=node_colors[node_name], 
                       ec='black', lw=2, zorder=2)
    ax1.add_patch(circle)
    ax1.text(x, y, node_name, ha='center', va='center', 
            fontsize=11, fontweight='bold', zorder=3)

ax1.set_title('A. Input Graph', fontsize=14, fontweight='bold', pad=15)

# Panel A (中) - UMAP
ax2 = plt.subplot(2, 3, 2)
for br in [0, 1, 2]:
    mask = branch == br
    ax2.scatter(z_umap[mask, 0], z_umap[mask, 1], 
               c=branch_colors[br], label=branch_labels[br], 
               alpha=0.6, s=15, edgecolors='none')
ax2.set_xlabel('UMAP 1', fontsize=11)
ax2.set_ylabel('UMAP 2', fontsize=11)
ax2.set_title('B. UMAP (colored by branch)', fontsize=14, fontweight='bold')
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)

# Panel A (右) - UMAP + Velocity Stream（展示分岔）
ax2_vel = plt.subplot(2, 3, 3)
if 'velocity_umap' in adata.obsm:
    # 使用有效的颜色参数：'branch' 或 'pseudotime'
    scv.pl.velocity_embedding_stream(adata, basis='umap', ax=ax2_vel, show=False,
                                     title='C. UMAP + Velocity Stream (showing bifurcation)',
                                     density=0.8, alpha=0.4, color='branch',
                                     arrow_size=2, frameon=False)
else:
    ax2_vel.text(0.5, 0.5, 'Velocity not computed', 
                ha='center', va='center', transform=ax2_vel.transAxes, fontsize=12)
    ax2_vel.set_title('C. UMAP + Velocity Stream', fontsize=14, fontweight='bold')

# Panel B (左) - Phase Portrait
ax3 = plt.subplot(2, 3, 4)
for br in [0, 1, 2]:
    mask = branch == br
    ax3.scatter(s_values[mask], u_values[mask], 
               c=branch_colors[br], label=branch_labels[br], 
               alpha=0.6, s=25, edgecolors='none')
ax3.set_xlabel('Spliced (s)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Unspliced (u)', fontsize=12, fontweight='bold')
ax3.set_title('D. Phase Portrait', fontsize=14, fontweight='bold')
ax3.legend(loc='best', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_aspect('equal', adjustable='box')

# Panel B (中) - UMAP + Velocity Arrows（更清晰地展示分岔）
ax4 = plt.subplot(2, 3, 5)
if 'velocity_umap' in adata.obsm:
    scv.pl.velocity_embedding(adata, basis='umap', ax=ax4, show=False,
                              title='E. UMAP + Velocity Arrows (bifurcation)',
                              arrow_size=2, density=0.5,
                              color='branch', frameon=False)
else:
    ax4.text(0.5, 0.5, 'Velocity not computed', 
            ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    ax4.set_title('E. UMAP + Velocity Arrows', fontsize=14, fontweight='bold')

# Panel B (右) - Velocity Graph（展示分岔点的连接）
ax5 = plt.subplot(2, 3, 6)
if 'velocity_graph' in adata.uns:
    scv.pl.velocity_embedding_stream(adata, basis='umap', ax=ax5, show=False,
                                     title='F. Velocity Graph (bifurcation structure)',
                                     density=1.0, alpha=0.6, color='pseudotime',
                                     frameon=False)
    # 高亮分岔点区域（Prog节点附近）
    prog_mask = (branch == 0) | (pseudotime < 0.3)
    ax5.scatter(z_umap[prog_mask, 0], z_umap[prog_mask, 1], 
               c='red', s=30, alpha=0.5, edgecolors='black', linewidths=0.5, 
               zorder=10, label='Branch Point')
    ax5.legend(loc='best', fontsize=9)
else:
    ax5.text(0.5, 0.5, 'Velocity graph not computed', 
            ha='center', va='center', transform=ax5.transAxes, fontsize=12)
    ax5.set_title('F. Velocity Graph', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(FIG_DIR / 'Fig1_Combined_with_Velocity.png', dpi=300, bbox_inches='tight')
plt.show()

# Panel C单独显示（因为尺寸不同）
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(t_series, u_series, 'o-', color=color_palette['deep_blue_gray'], label='Unspliced (u)', 
        linewidth=2.5, markersize=3, alpha=0.8)
ax.plot(t_series, s_series, 's-', color=color_palette['light_blue'], label='Spliced (s)', 
        linewidth=2.5, markersize=3, alpha=0.8)

ax.axvline(u_peak_time, color=color_palette['deep_blue_gray'], linestyle='--', alpha=0.5, linewidth=1.5)
ax.axvline(s_peak_time, color=color_palette['light_blue'], linestyle='--', alpha=0.5, linewidth=1.5)

ax.plot(u_peak_time, u_series[u_peak_idx], 'o', color=color_palette['deep_blue_gray'], 
        markersize=10, markeredgecolor='black', markeredgewidth=1.5)
ax.plot(s_peak_time, s_series[s_peak_idx], 's', color=color_palette['light_blue'], 
        markersize=10, markeredgecolor='black', markeredgewidth=1.5)

ax.text(u_peak_time, u_series[u_peak_idx] + 0.3, f'u peak\n(t={u_peak_time:.2f})', 
        ha='center', va='bottom', fontsize=10, color=color_palette['deep_blue_gray'], fontweight='bold')
ax.text(s_peak_time, s_series[s_peak_idx] + 0.3, f's peak\n(t={s_peak_time:.2f})', 
        ha='center', va='bottom', fontsize=10, color=color_palette['light_blue'], fontweight='bold')

ax.set_xlabel('Time (t)', fontsize=14, fontweight='bold')
ax.set_ylabel('Expression Level', fontsize=14, fontweight='bold')
ax.set_title('C. Time Series', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / 'Fig1_PanelC_standalone.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ 所有面板绘制完成并保存")
