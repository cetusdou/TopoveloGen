"""
Legacy notebook-export script. Prefer the panelized packages under experiments/figures/ (see experiments/figures/README.md).
"""


# ===== Original notebook cell 1 =====

# 导入必要的库
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path.cwd().parent if Path.cwd().name == 'figure_notebook' else Path.cwd()
sys.path.insert(0, str(project_root))
print(f"项目根目录: {project_root}")

from experiments.figure_style import apply_publication_style

apply_publication_style()

OUT_DIR = project_root / 'results' / 'fig_series_main1'
FIG_DIR = OUT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
print(f"图片输出目录: {FIG_DIR}")

# Replacement for notebook-local plot_utils.
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

def save_current_figure(stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.gcf()
    fig.savefig(FIG_DIR / f'{stem}.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / f'{stem}.svg', bbox_inches='tight')
    plt.close(fig)

# 尝试导入数据生成模块
try:
    from data_generation_module import GraphNode, CycleNode, GraphEdge, UniversalGraphSystem,SkeletonNode
    from data_generation_module import GraphSimulator
    from data_generation_module import GeneSynthesizer
    HAS_DATA_MODULE = True
except ImportError as e:
    print(f"警告: 无法导入数据生成模块: {e}")
    print("将使用模拟数据")
    HAS_DATA_MODULE = False

print("✓ 库导入完成")
print("✓ Publication style 已加载（styles/nature_science.mplstyle）")


# ===== Original notebook cell 2 =====

def generate_topology_data(topology_type, N_samples=1500, n_steps=80, dt=0.05, noise_level=0.15, seed=42):
    """
    生成不同拓扑结构的模拟数据
    
    Returns:
        topology_info: 拓扑信息字典，包含：
            - z_data: 最后一个时间点的潜在空间坐标 [N_samples, 2]
            - z_trajectory: 完整轨迹 [n_steps+1, N_samples, 2]
            - pseudotime: 伪时间 [N_samples]
            - branch: 分支标签 [N_samples]
            - time_points: 时间点 [n_steps+1]
    """
    np.random.seed(seed)
    np.random.seed(seed)


    # 使用真实的数据生成模块
    if topology_type == 'linear':
        nodes = [
            GraphNode("Start", [-2.0, -2.0], potential=-1.0, sigma=0.8),
            GraphNode("End", [2.0, 2.0], potential=-4.0, sigma=0.5)
        ]
        edges = [GraphEdge(nodes[0], nodes[1])]
        start_node = "Start"
        
    elif topology_type == 'bifurcation':
        nodes = [
            GraphNode("Stem", [-2.0, 0], potential=-1.0, sigma=0.8),
            GraphNode("Mid", [0.0, 0.0], potential=-2.0, sigma=0.5),
            GraphNode("FateA", [2.0, 1.5], potential=-5.0, sigma=0.8,basin_strength=0.5,basin_sigma=0.5),
            GraphNode("FateB", [2.0, -1.5], potential=-5.0, sigma=0.8,basin_strength=0.5,basin_sigma=0.5)
        ]
        edges = [
            GraphEdge(nodes[0], nodes[1]),
            GraphEdge(nodes[1], nodes[2]),
            GraphEdge(nodes[1], nodes[3])
        ]
        start_node = "Stem"
        
    elif topology_type == 'cycle':
        nodes = [
            GraphNode("Start", [0.0, 0], potential=0.0, sigma=0.8),
            CycleNode("Cycle", [1.0, 0.0], potential=-1.5, sigma=0.3, 
                        radius=1.0, omega=5.0, clockwise=False)
        ]
        edges = [GraphEdge(nodes[0], nodes[1])]
        start_node = "Start"
        
    elif topology_type == 'cycle_linear':
        nodes = [
            GraphNode("Start", [-2.0, 0], potential=-1.0, sigma=0.8),
            CycleNode("Cycle", [0.0, 0], potential=-3.0, sigma=0.5, 
                        radius=1.0, omega=4.0, clockwise=False),
            GraphNode("Exit", [2.0, 0], potential=-4.0, sigma=0.5)
        ]
        edges = [
            GraphEdge(nodes[0], nodes[1]),  # Start -> Cycle
            GraphEdge(nodes[1], nodes[2])   # Cycle -> Exit
        ]
        start_node = "Start"
        
    elif topology_type == 'figure8':
        # 使用 SkeletonNode 创建真正的8字形拓扑
        # 创建8字形状的骨架点
        t = np.linspace(0, 2 * np.pi, 300)  # 参数范围：0 到 2π
        x = 2.0 * np.sin(t)                  # x坐标
        y = 2.0 * np.sin(2 * t) / 2          # y坐标
        skeleton = np.stack([x, y], axis=1)  # 堆叠成 [300, 2] 的数组
        
        nodes = [
            GraphNode("Start", [-3.5, 0.0], potential=-1.0, sigma=0.8),
            SkeletonNode("8-cycle", skeleton_points=skeleton, 
                        potential=-3.5, sigma=0.35, flow_strength=5.0),
            GraphNode("End", [3.5, 0.0], potential=-5.5, sigma=0.5)
        ]
        edges = [
            GraphEdge(nodes[0], nodes[1]),  # Start -> Figure8 (自动使用最近邻停靠)
            GraphEdge(nodes[1], nodes[2])   # Figure8 -> End (自动使用最近邻停靠)
        ]
        start_node = "Start"
        
    elif topology_type == 'multifurcation':
        nodes = [
            GraphNode("Root", [-2.0, 0], potential=0.0, sigma=0.5),
            GraphNode("Branch1", [0.0, 1.5], potential=-3.0, sigma=0.8),
            GraphNode("Branch2", [0.0, 0], potential=-3.0, sigma=0.8),
            GraphNode("Branch3", [0.0, -1.5], potential=-3.0, sigma=0.8),
            GraphNode("Leaf1", [2.0, 2.0], potential=-5.0, sigma=0.5),
            GraphNode("Leaf2", [2.0, 0.5], potential=-5.0, sigma=0.5),
            GraphNode("Leaf3", [2.0, -0.5], potential=-5.0, sigma=0.5),
            GraphNode("Leaf4", [2.0, -2.0], potential=-5.0, sigma=0.5)
        ]
        edges = [
            GraphEdge(nodes[0], nodes[1]),
            GraphEdge(nodes[0], nodes[2]),
            GraphEdge(nodes[0], nodes[3]),
            GraphEdge(nodes[1], nodes[4]),
            GraphEdge(nodes[1], nodes[5]),
            GraphEdge(nodes[2], nodes[6]),
            GraphEdge(nodes[3], nodes[7])
        ]
        start_node = "Root"
    else:
        raise ValueError(f"Unknown topology type: {topology_type}")
    
    system = UniversalGraphSystem(nodes, edges, fusion_temp=0.05)
    traj = GraphSimulator.run(system, start_node, n_cells=N_samples, 
                                steps=n_steps, dt=dt, noise_level=noise_level)
    # traj.shape == (n_steps+1, N_samples, 2)
    # 确保是numpy数组并保留完整轨迹
    if isinstance(traj, np.ndarray):
        z_trajectory = traj
    else:
        z_trajectory = np.array(traj)
    z_data = z_trajectory[-1]  # 最后一个时间点的潜在空间坐标（用于计算pseudotime和branch）
    
    # 生成时间点
    time_points = np.arange(n_steps + 1) * dt
    
    # 使用真实时间作为pseudotime（所有细胞都在最后一个时间点采样）
    # 真实时间 = 最后一个时间点的时间
    real_time = time_points[-1]  # (n_steps) * dt
    pseudotime = np.full(N_samples, real_time)  # 所有细胞使用相同的真实时间
    
    # 计算branch（基于位置信息）
    if topology_type == 'linear':
        branch = np.zeros(N_samples, dtype=int)
    elif topology_type == 'bifurcation':
        branch = (z_data[:, 1] > 0).astype(int)
        branch[z_data[:, 1] <= 0] = 2
    elif topology_type == 'cycle':
        branch = np.zeros(N_samples, dtype=int)
    elif topology_type == 'cycle_linear':
        dist_to_origin = np.linalg.norm(z_data, axis=1)
        on_cycle = dist_to_origin < 1.3
        branch = (~on_cycle).astype(int)
    elif topology_type == 'figure8':
        # branch: 根据在8字的哪个"环"上（通过y坐标判断，上环为1，下环为2）
        branch = np.zeros(N_samples, dtype=int)
        branch[z_data[:, 1] > 0] = 1  # 上环
        branch[z_data[:, 1] <= 0] = 2  # 下环
    elif topology_type == 'multifurcation':
        branch = np.zeros(N_samples, dtype=int)
        branch[z_data[:, 1] > 0.3] = 1
        branch[z_data[:, 1] < -0.3] = 2
    
    # 直接使用原始潜在空间坐标，不进行UMAP降维
    topology_info = {
        'type': topology_type,
        'n_samples': N_samples,
        'z_data': z_data,  # 最后一个时间点的坐标
        'z_trajectory': z_trajectory,  # 完整轨迹 [n_steps+1, N_samples, 2]
        'pseudotime': pseudotime,
        'branch': branch,
        'time_points': time_points,  # 时间点 [n_steps+1]
        'n_steps': n_steps,
        'dt': dt,
        'system': system
    }
    
    return topology_info

print("✓ 拓扑生成函数定义完成")


# ===== Original notebook cell 3 =====

# 从完整轨迹中采样数据，模拟单细胞快照数据的特点
def sample_snapshot_data(z_trajectory, time_points, n_samples_per_time=None, random_seed=42):
    """
    从完整轨迹中采样数据点，模拟单细胞快照数据
    
    Args:
        z_trajectory: 完整轨迹 [n_steps+1, N_samples, 2]
        time_points: 时间点数组 [n_steps+1]
        n_samples_per_time: 每个时间步采样的细胞数（如果None，则均匀采样）
        random_seed: 随机种子
    
    Returns:
        sampled_z: 采样后的坐标 [total_samples, 2]
        sampled_time_labels: 每个采样点对应的时间标签（真实时间值）
    """
    np.random.seed(random_seed)
    n_steps, N_cells, _ = z_trajectory.shape
    
    if n_samples_per_time is None:
        # 均匀采样：从每个时间步采样相同数量的细胞
        # 确保总采样数约为原始细胞数
        n_samples_per_time = max(1, N_cells // n_steps)
    
    sampled_z_list = []
    sampled_time_labels = []
    
    for t_idx in range(n_steps):
        # 从当前时间步随机采样细胞
        n_sample_this_time = min(n_samples_per_time, N_cells)
        cell_indices = np.random.choice(N_cells, size=n_sample_this_time, replace=False)
        sampled_z_list.append(z_trajectory[t_idx, cell_indices])
        # 使用真实时间值作为标签
        time_label = time_points[t_idx]
        sampled_time_labels.extend([time_label] * len(cell_indices))
    
    sampled_z = np.vstack(sampled_z_list)
    sampled_time_labels = np.array(sampled_time_labels)
    
    return sampled_z, sampled_time_labels

print("✓ 采样函数定义完成")

# 计算速度场（从轨迹数据）
def compute_velocity_field(z_trajectory, time_points, grid_resolution=30):
    """
    从轨迹数据计算速度场
    
    Args:
        z_trajectory: 完整轨迹 [n_steps+1, N_samples, 2]
        time_points: 时间点数组 [n_steps+1]
        grid_resolution: 网格分辨率
    
    Returns:
        grid_x, grid_y: 网格坐标
        velocity_x, velocity_y: 速度场分量
    """
    n_timepoints, N_samples, _ = z_trajectory.shape
    n_steps = n_timepoints - 1  # 实际的时间步数（比时间点数少1）
    
    # 计算每个细胞在每个时间步的速度（有限差分）
    velocities = []
    positions = []
    
    for t_idx in range(n_steps):
        dt = time_points[t_idx + 1] - time_points[t_idx]
        if dt > 0:
            # 计算速度：v = dz/dt
            z_t = z_trajectory[t_idx]
            z_t_next = z_trajectory[t_idx + 1]
            v = (z_t_next - z_t) / dt
            velocities.append(v)
            positions.append(z_t)
    
    # 合并所有时间步的数据
    all_positions = np.vstack(positions)
    all_velocities = np.vstack(velocities)
    
    # 创建规则网格
    x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
    y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
    
    # 扩展边界
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= 0.1 * x_range
    x_max += 0.1 * x_range
    y_min -= 0.1 * y_range
    y_max += 0.1 * y_range
    
    grid_x = np.linspace(x_min, x_max, grid_resolution)
    grid_y = np.linspace(y_min, y_max, grid_resolution)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
    
    # 插值速度场到网格
    velocity_x = griddata(all_positions, all_velocities[:, 0], 
                         (grid_X, grid_Y), method='linear', fill_value=0)
    velocity_y = griddata(all_positions, all_velocities[:, 1], 
                         (grid_X, grid_Y), method='linear', fill_value=0)
    
    return grid_X, grid_Y, velocity_x, velocity_y

print("✓ 速度场计算函数定义完成")


# ===== Original notebook cell 4 =====

# 生成所有6种拓扑的数据
topology_types = ['linear', 'bifurcation', 'cycle', 'cycle_linear', 'figure8', 'multifurcation']
topology_labels = ['A. Linear', 'B. Bifurcation', 'C. Cycle', 
                   'D. Cycle-Linear', 'E. Figure-8', 'F. Multi-furcation']

topology_data = {}
print("生成拓扑数据...")
for i, topo_type in enumerate(topology_types):
    print(f"  {i+1}/6: {topology_labels[i]}...")
    topology_data[topo_type] = generate_topology_data(topo_type, N_samples=1500,n_steps=200, seed=42+i)
    print(f"    ✓ 完成 ({topology_data[topo_type]['n_samples']} 个细胞)")

print("\n✓ 所有拓扑数据生成完成")


# ===== Beautified plotting: FigSeriesMain1_TopologyCollection =====

from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize

# Embed fonts in vector exports (not covered by the mplstyle)
plt.rcParams.update({
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ---------- Visual constants ----------
TIME_CMAP = plt.get_cmap("TopoVelo_grad")
CAT_CMAP = plt.get_cmap("TopoVelo_cat")
STREAM_COLOR = "#5f6368"
EDGE_COLOR = "#8a8f94"
TEXT_COLOR = "#222222"
LIGHT_SPINE = "#b8b8b8"


def to_numpy(data):
    """Convert Tensor/List/array-like object to numpy array."""
    if hasattr(data, "detach"):
        data = data
    if hasattr(data, "cpu"):
        data = data
    if hasattr(data, "numpy"):
        data = data
    return np.asarray(data)


def pad_limits(x, y, pad=0.10):
    """Return padded equal-aspect limits."""
    x = np.asarray(x)
    y = np.asarray(y)
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    xr = max(xmax - xmin, 1e-6)
    yr = max(ymax - ymin, 1e-6)
    span = max(xr, yr)
    xc = 0.5 * (xmin + xmax)
    yc = 0.5 * (ymin + ymax)
    half = 0.5 * span * (1 + 2 * pad)
    return (xc - half, xc + half), (yc - half, yc + half)


def add_arrow_to_path(ax, x, y, n_arrows=2, color=EDGE_COLOR, size=8, lw=0.0, zorder=4):
    """Place small arrows along a path."""
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 3 or n_arrows <= 0:
        return
    idxs = np.linspace(1, len(x) - 2, n_arrows + 2, dtype=int)[1:-1]
    for i in idxs:
        p0 = (x[i], y[i])
        p1 = (x[i + 1], y[i + 1])
        ax.add_patch(FancyArrowPatch(
            p0, p1,
            arrowstyle="-|>",
            mutation_scale=size,
            color=color,
            lw=lw,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        ))


def draw_topology_schematic_beautified(ax, system, font_size=7.5):
    """Draw a compact schematic of the user-defined topological skeleton."""
    ax.set_aspect("equal")
    ax.axis("off")

    nodes = system.nodes
    edges = system.edges
    all_x, all_y = [], []

    # Edges first
    for edge in edges:
        start = to_numpy(edge.start)
        end = to_numpy(edge.end)
        x = np.linspace(start[0], end[0], 80)
        y = np.linspace(start[1], end[1], 80)
        ax.plot(x, y, color=EDGE_COLOR, lw=1.15, ls=(0, (2.2, 2.2)), zorder=1)
        add_arrow_to_path(ax, x, y, n_arrows=1, color=EDGE_COLOR, size=7, zorder=2)
        all_x.extend([start[0], end[0]])
        all_y.extend([start[1], end[1]])

    # Nodes
    n_nodes = max(len(nodes), 1)
    for idx, node in enumerate(nodes):
        color = CAT_CMAP(idx / max(n_nodes - 1, 1))
        pos = to_numpy(node.pos)

        if isinstance(node, CycleNode):
            theta = np.linspace(0, 2 * np.pi, 180)
            if getattr(node, "clockwise", False):
                theta = theta[::-1]
            x = pos[0] + node.radius * np.cos(theta)
            y = pos[1] + node.radius * np.sin(theta)
            ax.plot(x, y, color="white", lw=6.0, zorder=3)
            ax.plot(x, y, color=color, lw=3.0, zorder=4)
            add_arrow_to_path(ax, x, y, n_arrows=3, color=color, size=7, zorder=5)
            all_x.extend(x); all_y.extend(y)

        elif isinstance(node, SkeletonNode):
            skel = to_numpy(node.skeleton_points)
            x, y = skel[:, 0], skel[:, 1]
            ax.plot(x, y, color="white", lw=6.0, zorder=3)
            ax.plot(x, y, color=color, lw=3.0, zorder=4)
            add_arrow_to_path(ax, x, y, n_arrows=4, color=color, size=7, zorder=5)
            all_x.extend(x); all_y.extend(y)

        else:
            ax.scatter(
                pos[0], pos[1],
                s=82,
                facecolor=color,
                edgecolor="white",
                linewidth=1.0,
                zorder=5,
            )
            ax.scatter(
                pos[0], pos[1],
                s=88,
                facecolor="none",
                edgecolor="#333333",
                linewidth=0.45,
                zorder=6,
            )
            all_x.append(pos[0]); all_y.append(pos[1])

    if len(all_x) == 0:
        return
    y_span = max(all_y) - min(all_y)
    offset = 0.07 * y_span if y_span > 0 else 0.25

    for node in nodes:
        pos = to_numpy(node.pos)
        label_y = pos[1]
        if isinstance(node, CycleNode):
            label_y = pos[1] - 0.75 * node.radius
        elif isinstance(node, SkeletonNode):
            label_y = np.min(to_numpy(node.skeleton_points)[:, 1])
        ax.text(
            pos[0], label_y - offset, node.id,
            ha="center", va="top",
            fontsize=font_size,
            color=TEXT_COLOR,
            zorder=7,
        )

    (xlim, ylim) = pad_limits(np.array(all_x), np.array(all_y), pad=0.22)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def style_latent_axis(ax):
    """Minimal style for the latent dynamics panel."""
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def style_trace_axis(ax):
    """Small trace axis style aligned with nature_science.mplstyle."""
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.tick_params(labelsize=8, pad=1.5)


def plot_coordinate_trace(ax, sampled_time, sampled_vals, color,
                          ylabel=None, show_xlabel=False,
                          t_min=0, t_max=10):
    """Scatter of individual cell z-coordinate vs simulation time."""
    ax.scatter(sampled_time, sampled_vals,
               alpha=0.1, linewidth=0, s=5, color=color, rasterized=True)
    ax.set_xlim(t_min, t_max)
    y_min, y_max = sampled_vals.min(), sampled_vals.max()
    pad = 0.1 * (y_max - y_min + 1e-6)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_yticks([round(y_min, 1), round(y_max, 1)])
    ax.set_yticklabels([f"{y_min:.1f}", f"{y_max:.1f}"], fontsize=7)
    ax.set_ylabel(ylabel or "", labelpad=2)
    if show_xlabel:
        ax.set_xlabel("Simulation Time", labelpad=1.5)
    else:
        ax.tick_params(labelbottom=False)
    style_trace_axis(ax)


def make_beautified_topology_collection(save_stem="FigSeriesMain1_TopologyCollection"):
    """Create the refined 6 x 3 topology collection figure."""
    n_rows = len(topology_types)
    fig = plt.figure(figsize=(7.2, 10.4), dpi=300, constrained_layout=False)
    gs = fig.add_gridspec(
        n_rows, 3,
        width_ratios=[0.88, 1.26, 1.22],
        hspace=0.34,
        wspace=0.18,
        left=0.055,
        right=0.955,
        top=0.935,
        bottom=0.055,
    )

    # Column headers
    header_y = 0.972
    fig.text(0.145, header_y, "Topology", ha="center", va="center", fontsize=9, fontweight="bold")
    fig.text(0.440, header_y, "Latent dynamics", ha="center", va="center", fontsize=9, fontweight="bold")
    fig.text(0.765, header_y, "Latent coordinate traces", ha="center", va="center", fontsize=9, fontweight="bold")

    t_min = min(topology_data[t]["time_points"][0] for t in topology_types)
    t_max = max(topology_data[t]["time_points"][-1] for t in topology_types)
    norm = Normalize(vmin=t_min, vmax=t_max)

    for idx, topo_type in enumerate(topology_types):
        data = topology_data[topo_type]
        title_text = topology_labels[idx].split(". ", 1)[1]
        z_trajectory = data["z_trajectory"]
        time_points = data["time_points"]

        sampled_z, sampled_time_labels = sample_snapshot_data(
            z_trajectory, time_points,
            n_samples_per_time=60,
            random_seed=42 + idx,
        )

        t_final = time_points[-1]

        # Topology schematic
        ax_topo = fig.add_subplot(gs[idx, 0])
        draw_topology_schematic_beautified(ax_topo, data["system"], font_size=6.8)
        ax_topo.text(
            -0.05, 1.02, chr(ord("A") + idx),
            transform=ax_topo.transAxes,
            ha="right", va="bottom",
            fontsize=13,
            fontweight="bold",
            color=TEXT_COLOR,
        )

        # Latent dynamics panel
        ax_dyn = fig.add_subplot(gs[idx, 1])
        grid_X, grid_Y, velocity_x, velocity_y = compute_velocity_field(
            z_trajectory, time_points, grid_resolution=34
        )
        ax_dyn.streamplot(
            grid_X, grid_Y, velocity_x, velocity_y,
            density=1.05,
            color=STREAM_COLOR,
            linewidth=0.42,
            arrowsize=0.58,
            arrowstyle="->",
            minlength=0.12,
            zorder=1,
        )
        # Trajectory colored by simulation time (consistent with colorbar)
        ax_dyn.scatter(
            sampled_z[:, 0], sampled_z[:, 1],
            c=sampled_time_labels,
            cmap=TIME_CMAP,
            norm=norm,
            s=3.5,
            alpha=0.55,
            linewidth=0,
            rasterized=True,
            zorder=3,
        )
        xlim, ylim = pad_limits(z_trajectory[:, :, 0].ravel(), z_trajectory[:, :, 1].ravel(), pad=0.07)
        ax_dyn.set_xlim(*xlim)
        ax_dyn.set_ylim(*ylim)
        ax_dyn.set_title(title_text, pad=2.5, color=TEXT_COLOR)
        style_latent_axis(ax_dyn)

        # Coordinate traces (scatter: sampled_time vs z1/z2)
        gs_trace = gs[idx, 2].subgridspec(2, 1, hspace=0, height_ratios=[1, 1])
        ax_z1 = fig.add_subplot(gs_trace[0])
        ax_z2 = fig.add_subplot(gs_trace[1], sharex=ax_z1)
        plot_coordinate_trace(
            ax_z1, sampled_time_labels, sampled_z[:, 0],
            color=CAT_CMAP(0.1), ylabel=r"$z_1$", show_xlabel=False,
            t_min=time_points[0], t_max=time_points[-1],
        )
        plot_coordinate_trace(
            ax_z2, sampled_time_labels, sampled_z[:, 1],
            color=CAT_CMAP(0.9), ylabel=r"$z_2$",
            show_xlabel=(idx == n_rows - 1),
            t_min=time_points[0], t_max=time_points[-1],
        )

    # Shared colorbar
    cax = fig.add_axes([0.965, 0.18, 0.012, 0.62])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=TIME_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Simulation time", labelpad=4)
    cbar.ax.tick_params(labelsize=8, length=3.5)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=CAT_CMAP(0.1), lw=0, marker="o",
               markersize=4, alpha=0.6, label=r"$z_1$ cells"),
        Line2D([0], [0], color=CAT_CMAP(0.9), lw=0, marker="o",
               markersize=4, alpha=0.6, label=r"$z_2$ cells"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.95, 0.028),
        ncol=2,
        handlelength=1.0,
        columnspacing=1.0,
    )

    save_current_figure(save_stem)
    return fig


fig = make_beautified_topology_collection()
print("✓ Beautified topology collection figure saved")
