import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List

# 通过相对导入获取类型帮助（运行时不会强制依赖）
from .graph_core import UniversalGraphSystem, GraphNode, CycleNode, GraphEdge
class GraphSimulator:
    @staticmethod
    def run(system, start_node_id, n_cells=50, steps=1000, dt=0.01, noise_level=0.1):
        try:
            start_node = next(n for n in system.nodes if n.id == start_node_id)
        except StopIteration:
            raise ValueError(f"Start node {start_node_id} not found")
            
        dim = len(start_node.pos)
        start_pos = torch.tensor(start_node.pos, dtype=torch.float32)
        
        z = start_pos.unsqueeze(0).repeat(n_cells, 1)
        z += torch.randn_like(z) * start_node.sigma *0.1
        
        traj = [z.clone()]
        
        print(f"Starting simulation in {dim}-Dimensional space...")
        for i in range(steps):
            drift = system.get_drift(z)
            noise = torch.randn_like(z) * noise_level * np.sqrt(dt)
            z = z + drift * dt + noise
            traj.append(z.clone())
            
        return torch.stack(traj).detach().cpu().numpy()

class GraphVisualizer:
    """
    高级可视化工具 (适配 CycleNode 和混合动力学)
    """
    @staticmethod
    def plot_landscape(system,
                       title: str = "Potential Landscape",
                       bounds: Tuple[float, float] = (-3.5, 3.5),
                       show_skeleton: bool = True,
                       stream_threshold: float = -0.01,
                       stride: int = 5,
                       ax: Optional[plt.Axes] = None,
                       show: bool = True,
                       **plot_kwargs):
        """
        Plot potential landscape with many configurable options via `plot_kwargs`.

        Supported `plot_kwargs` (defaults set to preserve previous behaviour):
        - res: int (grid resolution, default 150)
        - cmap: str or Colormap (default 'viridis_r')
        - levels_count: int (number of contour levels, default 60)
        - stream_stride: int (sparse stride for streamplot, default `stride`)
        - stream_density: float (streamplot density, default 1.0)
        - stream_color: color (default 'white')
        - stream_linewidth: float (default 0.6)
        - arrowsize: float (default 0.8)
        - skeleton_edge_style: dict (kwargs forwarded to `ax.plot` for skeleton edges)
        - cycle_style: dict (circle style for CycleNode)
        - node_style: dict (style for GraphNode markers/text)
        - show_colorbar: bool (whether to show colorbar, default True)
        """
        
        # 1. 生成高分辨率网格 (用于背景势能 Contourf)
        limit = bounds[1]
        res = int(plot_kwargs.get('res', 150))
        x = np.linspace(-limit, limit, res)
        y = np.linspace(-limit, limit, res)
        X, Y = np.meshgrid(x, y)
        grid = torch.tensor(np.stack([X, Y], axis=-1), dtype=torch.float32).reshape(-1, 2)
        
        # 计算高分率势能
        U = system.potential_numpy(grid).reshape(res, res)
        
        # 创建/复用画布
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=plot_kwargs.get('figsize', (11, 9)))
            created_fig = True
        else:
            fig = ax.figure
        
        # 2. 绘制势能背景
        v_min, v_max = U.min(), np.percentile(U, 95)
        # 扩展 levels 以便背景平滑
        levels_count = int(plot_kwargs.get('levels_count', 60))
        levels = np.linspace(v_min, max(v_max, 0.1), levels_count)
        cmap = plot_kwargs.get('cmap', 'viridis_r')
        cntr = ax.contourf(X, Y, U, levels=levels, cmap=cmap, extend='both')
        show_colorbar = bool(plot_kwargs.get('show_colorbar', True))
        if show_colorbar:
            # attach colorbar to the figure/axis in use
            cbar = fig.colorbar(cntr, ax=ax)
            cbar.set_label(plot_kwargs.get('colorbar_label', "Potential Energy (Depth)"), rotation=270, labelpad=15)
        
        # 3. 绘制局部流线 (Streamplot)
        # 使用稀疏网格
        stream_stride = int(plot_kwargs.get('stream_stride', stride))
        x_s = x[::stream_stride]; y_s = y[::stream_stride]
        X_s, Y_s = np.meshgrid(x_s, y_s)
        grid_s = torch.tensor(np.stack([X_s, Y_s], axis=-1), dtype=torch.float32).reshape(-1, 2)
        
        # A. 获取漂移速度
        with torch.enable_grad():
            drift_s = system.get_drift(grid_s).detach().numpy()
            
        U_s = drift_s[:, 0].reshape(X_s.shape)
        V_s = drift_s[:, 1].reshape(Y_s.shape)
        
        pot_s = system.potential_numpy(grid_s).reshape(X_s.shape)
        mask = pot_s > stream_threshold
        
        # 将背景区域的速度设为 NaN，streamplot 会自动跳过
        U_s[mask] = np.nan
        V_s[mask] = np.nan
        
        # 绘制流线
        stream_color = plot_kwargs.get('stream_color', 'white')
        stream_linewidth = float(plot_kwargs.get('stream_linewidth', 0.6))
        stream_density = float(plot_kwargs.get('stream_density', 1.0))
        arrowsize = float(plot_kwargs.get('arrowsize', 0.8))
        ax.streamplot(X_s, Y_s, U_s, V_s, color=stream_color, linewidth=stream_linewidth,
                  density=stream_density, arrowsize=arrowsize)
        
        # 4. 绘制图骨架 (Skeleton)
        if show_skeleton:
            edge_style = plot_kwargs.get('skeleton_edge_style', {'c': 'white', 'ls': ':', 'lw': 1.0, 'alpha': 0.4})
            for edge in system.edges:
                start, end = edge.start.cpu().numpy(), edge.end.cpu().numpy()
                ax.plot([start[0], end[0]], [start[1], end[1]], zorder=2, **edge_style)

            cycle_style = plot_kwargs.get('cycle_style', {'color': 'cyan', 'fill': False, 'ls': '--', 'linewidth': 1.5, 'alpha': 0.6})
            node_text_kwargs = plot_kwargs.get('node_text_kwargs', {})
            for node in system.nodes:
                if hasattr(node, 'radius'):  # CycleNode
                    circle = plt.Circle(node.pos, node.radius, **cycle_style, zorder=3)
                    ax.add_artist(circle)

                    direction_sym = "↻" if node.clockwise else "↺"
                    txt = f"{node.id}\n{direction_sym}\nω={node.omega}"
                    ax.text(node.pos[0], node.pos[1], txt, color=cycle_style.get('color', 'cyan'),
                            fontsize=plot_kwargs.get('cycle_text_fontsize', 9), fontweight='bold',
                            ha='center', va='center', zorder=4, **node_text_kwargs)
                else:  # GraphNode
                    if getattr(node, 'basin_strength', 0) > 0:
                        basin_style = plot_kwargs.get('basin_circle_style', {'color': 'red', 'fill': False, 'ls': ':', 'alpha': 0.5})
                        circle = plt.Circle(node.pos, node.basin_sigma, **basin_style, zorder=3)
                        ax.add_artist(circle)

                    ax.text(node.pos[0], node.pos[1], f"{node.id}", color=plot_kwargs.get('node_label_color', 'white'),
                            fontsize=plot_kwargs.get('node_label_fontsize', 9), fontweight='bold', ha='center', va='center', zorder=4)

        ax.set_title(title, fontsize=plot_kwargs.get('title_fontsize', 14))
        ax.set_xlabel("Latent Dim 1")
        ax.set_ylabel("Latent Dim 2")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        plt.tight_layout()
        if show:
            plt.show()

        return fig, ax


    @staticmethod
    def plot_traj(system,
                  traj: np.ndarray = None,
                  title: str = "Potential Landscape",
                  bounds: Tuple[float, float] = (-3.5, 3.5),
                  show_skeleton: bool = True,
                  stream_threshold: float = -0.01,
                  stride: int = 5,
                  ax: Optional[plt.Axes] = None,
                  show: bool = True,
                  **plot_kwargs):
        """
        Plot landscape + trajectories. Accepts same `plot_kwargs` as `plot_landscape`.
        """
        
        # 1. 生成高分辨率网格 (用于背景势能 Contourf)
        limit = bounds[1]
        res = int(plot_kwargs.get('res', 150))
        x = np.linspace(-limit, limit, res)
        y = np.linspace(-limit, limit, res)
        X, Y = np.meshgrid(x, y)
        grid = torch.tensor(np.stack([X, Y], axis=-1), dtype=torch.float32).reshape(-1, 2)
        
        # 计算高分率势能
        U = system.potential_numpy(grid).reshape(res, res)
        
        # 创建/复用画布
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=plot_kwargs.get('figsize', (11, 9)))
            created_fig = True
        else:
            fig = ax.figure
        
        # 2. 绘制势能背景
        v_min, v_max = U.min(), np.percentile(U, 95)
        # 扩展 levels 以便背景平滑
        levels_count = int(plot_kwargs.get('levels_count', 60))
        levels = np.linspace(v_min, max(v_max, 0.1), levels_count)
        cmap = plot_kwargs.get('cmap', 'viridis_r')
        cntr = ax.contourf(X, Y, U, levels=levels, cmap=cmap, extend='both')
        show_colorbar = bool(plot_kwargs.get('show_colorbar', True))
        if show_colorbar:
            cbar = fig.colorbar(cntr, ax=ax)
            cbar.set_label(plot_kwargs.get('colorbar_label', "Potential Energy (Depth)"), rotation=270, labelpad=15)
        
        # 3. [关键修改] 绘制局部流线 (Streamplot)
        # 使用稀疏网格
        stream_stride = int(plot_kwargs.get('stream_stride', stride))
        x_s = x[::stream_stride]; y_s = y[::stream_stride]
        X_s, Y_s = np.meshgrid(x_s, y_s)
        grid_s = torch.tensor(np.stack([X_s, Y_s], axis=-1), dtype=torch.float32).reshape(-1, 2)
        
        # A. 获取漂移速度
        with torch.enable_grad():
            drift_s = system.get_drift(grid_s).detach().numpy()
            
        U_s = drift_s[:, 0].reshape(X_s.shape)
        V_s = drift_s[:, 1].reshape(Y_s.shape)
        
        # B. [新增步骤] 计算稀疏网格的势能，用于掩码
        # 注意：必须重新计算 grid_s 的势能，不能直接从高分辨率 U 中切片，那样容易对不齐
        pot_s = system.potential_numpy(grid_s).reshape(X_s.shape)
        
        # C. [核心逻辑] 应用掩码
        # 我们的沟渠势能是负数 (例如 -2.0)。背景是 0.0。
        # 如果势能 > stream_threshold (比如 -0.05)，说明太浅了/在背景里，不画流线。
        mask = pot_s > stream_threshold
        
        # 将背景区域的速度设为 NaN，streamplot 会自动跳过
        U_s[mask] = np.nan
        V_s[mask] = np.nan
        
        # 绘制流线
        stream_color = plot_kwargs.get('stream_color', 'white')
        stream_linewidth = float(plot_kwargs.get('stream_linewidth', 0.6))
        stream_density = float(plot_kwargs.get('stream_density', 1.0))
        arrowsize = float(plot_kwargs.get('arrowsize', 0.8))
        ax.streamplot(X_s, Y_s, U_s, V_s, color=stream_color, linewidth=stream_linewidth,
                  density=stream_density, arrowsize=arrowsize)
        
        # 4. 绘制图骨架 (Skeleton)
        if show_skeleton:
            edge_style = plot_kwargs.get('skeleton_edge_style', {'c': 'white', 'ls': ':', 'lw': 1.0, 'alpha': 0.4})
            for edge in system.edges:
                start, end = edge.start.cpu().numpy(), edge.end.cpu().numpy()
                ax.plot([start[0], end[0]], [start[1], end[1]], zorder=2, **edge_style)

            cycle_style = plot_kwargs.get('cycle_style', {'color': 'cyan', 'fill': False, 'ls': '--', 'linewidth': 1.5, 'alpha': 0.6})
            node_text_kwargs = plot_kwargs.get('node_text_kwargs', {})
            for node in system.nodes:
                if hasattr(node, 'radius'):  # CycleNode
                    circle = plt.Circle(node.pos, node.radius, **cycle_style, zorder=3)
                    ax.add_artist(circle)

                    direction_sym = "↻" if node.clockwise else "↺"
                    txt = f"{node.id}\n{direction_sym}\nω={node.omega}"
                    ax.text(node.pos[0], node.pos[1], txt, color=cycle_style.get('color', 'cyan'),
                            fontsize=plot_kwargs.get('cycle_text_fontsize', 9), fontweight='bold',
                            ha='center', va='center', zorder=4, **node_text_kwargs)
                else:  # GraphNode
                    if getattr(node, 'basin_strength', 0) > 0:
                        basin_style = plot_kwargs.get('basin_circle_style', {'color': 'red', 'fill': False, 'ls': ':', 'alpha': 0.5})
                        circle = plt.Circle(node.pos, node.basin_sigma, **basin_style, zorder=3)
                        ax.add_artist(circle)

                    ax.text(node.pos[0], node.pos[1], f"{node.id}", color=plot_kwargs.get('node_label_color', 'white'),
                            fontsize=plot_kwargs.get('node_label_fontsize', 9), fontweight='bold', ha='center', va='center', zorder=4)

        # 5. 绘制模拟轨迹
        if traj is not None:
            n_plot = min(100, traj.shape[1])
            for i in range(n_plot):
                ax.plot(traj[:, i, 0], traj[:, i, 1], c='white', lw=0.5, alpha=0.3, zorder=1)
                
        ax.set_title(title, fontsize=plot_kwargs.get('title_fontsize', 14))
        ax.set_xlabel("Latent Dim 1")
        ax.set_ylabel("Latent Dim 2")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        plt.tight_layout()
        if show:
            plt.show()

        return fig, ax
