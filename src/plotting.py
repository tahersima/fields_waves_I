# -*- coding: utf-8 -*-
# By: Mohammad H. Tahersima
# At: 2026-09-04
# ALL RIGHTS RESERVED
"""Scientific-grade, publication-ready plotting module for electromagnetic fields.

Designed for undergraduate electromagnetics education (Spyder-compatible).
Provides reusable styling and visualizations:
  1. 3D spatial distribution of electric field vectors and charge markers.
  2. 2D cross-section planar slices (xy, xz, and yz planes) with contours & quivers.
  3. Combined 2x2 multi-panel publication summary figure.
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Literal
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)


# ==============================================================================
# Publication Styling Configuration
# ==============================================================================
def apply_publication_style() -> None:
    """Apply clean, high-contrast, publication-grade styling to Matplotlib."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": ":",
        "figure.autolayout": False,
        "mathtext.fontset": "cm",  # Computer Modern LaTeX math font
    })


apply_publication_style()


# ==============================================================================
# Internal Helper Functions
# ==============================================================================
def _draw_charges_2d(
    ax: plt.Axes, 
    charges: List[dict], 
    u_idx: int, 
    v_idx: int,
    u_label: str,
    v_label: str
) -> None:
    """Plot charge positions on a 2D cross-section."""
    for i, chg in enumerate(charges):
        pos = np.array(chg["pos"], dtype=float)
        q = float(chg["q"])
        color = "#d62728" if q > 0 else "#1f77b4"  # Red for (+), Blue for (-)
        symbol = "+" if q > 0 else "−"
        
        # Scatter charge point
        ax.scatter(
            pos[u_idx], pos[v_idx], 
            color=color, s=140, edgecolors="black", linewidths=1.2, zorder=10
        )
        ax.text(
            pos[u_idx], pos[v_idx], symbol, 
            color="white", ha="center", va="center", fontsize=11, fontweight="bold", zorder=11
        )
        ax.text(
            pos[u_idx], pos[v_idx] + 0.08, f"$q_{{{i+1}}}={q:+.1e}\\,\\mathrm{{C}}$", 
            color="black", ha="center", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
            zorder=12
        )


def _draw_charges_3d(ax: Axes3D, charges: List[dict]) -> None:
    """Plot charge positions in a 3D axes."""
    for i, chg in enumerate(charges):
        pos = np.array(chg["pos"], dtype=float)
        q = float(chg["q"])
        color = "#d62728" if q > 0 else "#1f77b4"
        symbol = "+" if q > 0 else "−"
        
        ax.scatter(
            [pos[0]], [pos[1]], [pos[2]], 
            color=color, s=160, edgecolors="black", linewidths=1.2, depthshade=False, zorder=10
        )
        ax.text(
            pos[0], pos[1], pos[2], f"  {symbol} $q_{{{i+1}}}$", 
            color="black", fontsize=9, fontweight="bold", zorder=11
        )


# ==============================================================================
# 1. Standalone 3D Field Visualization
# ==============================================================================
def plot_field_3d(
    x_mesh: npt.NDArray[np.float64],
    y_mesh: npt.NDArray[np.float64],
    z_mesh: npt.NDArray[np.float64],
    Ex: npt.NDArray[np.float64],
    Ey: npt.NDArray[np.float64],
    Ez: npt.NDArray[np.float64],
    charges: List[dict],
    show_vectors: bool = True,
    target_vectors_per_axis: int = 10,
    vector_mode: Literal["volume", "cross_planes"] = "volume",
    slice_planes: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    title: str = "3D Electric Field Distribution $\\mathbf{E}(\\mathbf{r})$"
) -> plt.Figure:
    """Create a publication-grade standalone 3D visualization of the electric field.
    
    Args:
        x_mesh, y_mesh, z_mesh: 3D coordinate meshgrid arrays (meters).
        Ex, Ey, Ez: Vector components of electric field (V/m).
        charges: List of dicts with 'pos' (tuple/list of 3 floats) and 'q' (float in Coulombs).
        show_vectors: If True, renders vector quiver arrows.
        target_vectors_per_axis: Target number of arrows per dimension (decouples visual density from grid resolution).
        vector_mode: 'volume' (clean, sparse 3D grid) or 'cross_planes' (unobstructed orthogonal cuts).
        slice_planes: (x_cut, y_cut, z_cut) coordinates used when vector_mode='cross_planes'.
        title: Figure title.
    """
    fig = plt.figure(figsize=(9, 7), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    
    # Decouple visual arrow sampling from the fine computational resolution
    stride = max(1, x_mesh.shape[0] // target_vectors_per_axis)
    
    # 3D Vector Quiver
    if show_vectors:
        if vector_mode == "cross_planes":
            # Extract arrows strictly on the orthogonal coordinate cut planes to keep interior visible
            iz = int(np.argmin(np.abs(z_mesh[0, 0, :] - slice_planes[2])))
            iy = int(np.argmin(np.abs(y_mesh[0, :, 0] - slice_planes[1])))
            ix = int(np.argmin(np.abs(x_mesh[:, 0, 0] - slice_planes[0])))
            
            sub_step = max(1, x_mesh.shape[0] // (target_vectors_per_axis * 2))
            
            xs_list, ys_list, zs_list = [], [], []
            u_list, v_list, w_list = [], [], []
            
            # Plane 1: z = slice_planes[2]
            xs_list.append(x_mesh[::sub_step, ::sub_step, iz].flatten())
            ys_list.append(y_mesh[::sub_step, ::sub_step, iz].flatten())
            zs_list.append(z_mesh[::sub_step, ::sub_step, iz].flatten())
            u_list.append(Ex[::sub_step, ::sub_step, iz].flatten())
            v_list.append(Ey[::sub_step, ::sub_step, iz].flatten())
            w_list.append(Ez[::sub_step, ::sub_step, iz].flatten())
            
            # Plane 2: y = slice_planes[1]
            xs_list.append(x_mesh[::sub_step, iy, ::sub_step].flatten())
            ys_list.append(y_mesh[::sub_step, iy, ::sub_step].flatten())
            zs_list.append(z_mesh[::sub_step, iy, ::sub_step].flatten())
            u_list.append(Ex[::sub_step, iy, ::sub_step].flatten())
            v_list.append(Ey[::sub_step, iy, ::sub_step].flatten())
            w_list.append(Ez[::sub_step, iy, ::sub_step].flatten())
            
            xs = np.concatenate(xs_list)
            ys = np.concatenate(ys_list)
            zs = np.concatenate(zs_list)
            u = np.concatenate(u_list)
            v = np.concatenate(v_list)
            w = np.concatenate(w_list)
        else:
            # Sparse 3D volume grid (~10 points per axis, regardless of whether mesh resolution is 30 or 200+)
            xs = x_mesh[::stride, ::stride, ::stride].flatten()
            ys = y_mesh[::stride, ::stride, ::stride].flatten()
            zs = z_mesh[::stride, ::stride, ::stride].flatten()
            u = Ex[::stride, ::stride, ::stride].flatten()
            v = Ey[::stride, ::stride, ::stride].flatten()
            w = Ez[::stride, ::stride, ::stride].flatten()
        
        mags = np.sqrt(u**2 + v**2 + w**2)
        safe_mags = np.where(mags == 0, 1.0, mags)
        u_norm = u / safe_mags
        v_norm = v / safe_mags
        w_norm = w / safe_mags
        
        # Color arrows by log-magnitude for visual dynamic range
        cmap = plt.cm.plasma
        vmin = float(max(np.percentile(mags, 5), 1e-3))
        vmax = float(max(np.percentile(mags, 98), vmin * 10))
        norm = LogNorm(vmin=vmin, vmax=vmax)
        colors = cmap(norm(np.clip(mags, vmin, vmax)))
        
        # Dynamic arrow sizing scaled to actual spacing between sampled points
        grid_span = float(x_mesh.max() - x_mesh.min())
        arrow_length = (grid_span / target_vectors_per_axis) * 0.55
        
        ax.quiver(
            xs, ys, zs, u_norm, v_norm, w_norm,
            length=arrow_length, normalize=False, colors=colors,
            arrow_length_ratio=0.35, linewidth=0.8, alpha=0.65
        )
        
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1, aspect=18)
        cbar.set_label(r"$|\mathbf{E}|\;(\mathrm{V/m})$", rotation=270, labelpad=15)

    _draw_charges_3d(ax, charges)
    
    ax.set_xlabel(r"$x\;(\mathrm{m})$", labelpad=8)
    ax.set_ylabel(r"$y\;(\mathrm{m})$", labelpad=8)
    ax.set_zlabel(r"$z\;(\mathrm{m})$", labelpad=8)
    ax.set_title(title, pad=15)
    
    fig.tight_layout()
    return fig


# ==============================================================================
# 2. Standalone 2D Cross-Section Slice Visualization
# ==============================================================================
def plot_field_2d_slice(
    x_mesh: npt.NDArray[np.float64],
    y_mesh: npt.NDArray[np.float64],
    z_mesh: npt.NDArray[np.float64],
    Ex: npt.NDArray[np.float64],
    Ey: npt.NDArray[np.float64],
    Ez: npt.NDArray[np.float64],
    charges: List[dict],
    plane: Literal["xy", "xz", "yz"] = "xy",
    slice_coord: float = 0.0,
    show_vectors: bool = True,
    quiver_density: int = 18,
    title: Optional[str] = None
) -> plt.Figure:
    """Create a publication-grade 2D planar slice showing field magnitude & vectors.
    
    Args:
        plane: Planar slice to extract ('xy', 'xz', or 'yz').
        slice_coord: Value along the normal axis at which to take the cut (default 0.0 m).
        show_vectors: Whether to overlay vector quiver arrows.
        quiver_density: Number of arrow points along each axis.
    """
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    
    # Extract planar slice
    if plane == "xy":
        normal_coords = z_mesh[0, 0, :]
        idx = int(np.argmin(np.abs(normal_coords - slice_coord)))
        actual_val = normal_coords[idx]
        u_grid = x_mesh[:, :, idx]
        v_grid = y_mesh[:, :, idx]
        eu = Ex[:, :, idx]
        ev = Ey[:, :, idx]
        ew = Ez[:, :, idx]
        u_idx, v_idx = 0, 1
        u_label, v_label = r"$x\;(\mathrm{m})$", r"$y\;(\mathrm{m})$"
        cut_label = f"$z = {actual_val:.2f}\\,\\mathrm{{m}}$"
    elif plane == "xz":
        normal_coords = y_mesh[0, :, 0]
        idx = int(np.argmin(np.abs(normal_coords - slice_coord)))
        actual_val = normal_coords[idx]
        u_grid = x_mesh[:, idx, :]
        v_grid = z_mesh[:, idx, :]
        eu = Ex[:, idx, :]
        ev = Ez[:, idx, :]
        ew = Ey[:, idx, :]
        u_idx, v_idx = 0, 2
        u_label, v_label = r"$x\;(\mathrm{m})$", r"$z\;(\mathrm{m})$"
        cut_label = f"$y = {actual_val:.2f}\\,\\mathrm{{m}}$"
    elif plane == "yz":
        normal_coords = x_mesh[:, 0, 0]
        idx = int(np.argmin(np.abs(normal_coords - slice_coord)))
        actual_val = normal_coords[idx]
        u_grid = y_mesh[idx, :, :]
        v_grid = z_mesh[idx, :, :]
        eu = Ey[idx, :, :]
        ev = Ez[idx, :, :]
        ew = Ex[idx, :, :]
        u_idx, v_idx = 1, 2
        u_label, v_label = r"$y\;(\mathrm{m})$", r"$z\;(\mathrm{m})$"
        cut_label = f"$x = {actual_val:.2f}\\,\\mathrm{{m}}$"
    else:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")

    E_mag = np.sqrt(eu**2 + ev**2 + ew**2)
    # Clip extreme singularities at point charges for smooth publication color mapping
    vmin = float(max(np.percentile(E_mag, 5), 1e-3))
    vmax = float(max(np.percentile(E_mag, 98), vmin * 10))
    norm = LogNorm(vmin=vmin, vmax=vmax)
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 50)

    # Continuous field contour
    contour = ax.contourf(
        u_grid, v_grid, E_mag, levels=levels, cmap="plasma", norm=norm, extend="both"
    )
    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$|\mathbf{E}|\;(\mathrm{V/m})$", rotation=270, labelpad=15)
    
    # In-plane vector quivers
    if show_vectors:
        step_u = max(1, u_grid.shape[0] // quiver_density)
        step_v = max(1, v_grid.shape[1] // quiver_density)
        
        qu = u_grid[::step_u, ::step_v]
        qv = v_grid[::step_u, ::step_v]
        q_eu = eu[::step_u, ::step_v]
        q_ev = ev[::step_u, ::step_v]
        
        in_plane_mag = np.hypot(q_eu, q_ev)
        safe_in_plane = np.where(in_plane_mag == 0, 1.0, in_plane_mag)
        
        # Directional arrows with constant visual size, colored white for high contrast
        ax.quiver(
            qu, qv, q_eu / safe_in_plane, q_ev / safe_in_plane,
            color="white", scale=30, width=0.0035, headwidth=4, headlength=4.5,
            alpha=0.85, label=r"$\mathbf{E}_\mathrm{plane}$ direction"
        )
        ax.legend(loc="upper right", framealpha=0.85)

    _draw_charges_2d(ax, charges, u_idx, v_idx, u_label, v_label)
    
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(u_label)
    ax.set_ylabel(v_label)
    ax.set_title(title or f"Electric Field in {plane.upper()} Plane ({cut_label})")
    
    fig.tight_layout()
    return fig


# ==============================================================================
# 3. Combined 2x2 Multi-Panel Publication Summary Figure
# ==============================================================================
def plot_field_2x2_summary(
    x_mesh: npt.NDArray[np.float64],
    y_mesh: npt.NDArray[np.float64],
    z_mesh: npt.NDArray[np.float64],
    Ex: npt.NDArray[np.float64],
    Ey: npt.NDArray[np.float64],
    Ez: npt.NDArray[np.float64],
    charges: List[dict],
    show_vectors: bool = True,
    slice_coords: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    title: str = "Coulomb Electrostatic Field & Spatial Distribution Analysis"
) -> plt.Figure:
    """Create a unified 2x2 publication summary:
      - Panel (a): 3D Global Perspective
      - Panel (b): XY Cross-Section (top view)
      - Panel (c): XZ Cross-Section (side view)
      - Panel (d): YZ Cross-Section (end view)
    """
    fig = plt.figure(figsize=(14, 12), dpi=150)
    
    E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    vmin = float(max(np.percentile(E_mag, 5), 1e-3))
    vmax = float(max(np.percentile(E_mag, 98), vmin * 10))
    norm = LogNorm(vmin=vmin, vmax=vmax)
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 45)
    
    # ---------------- Panel (a): 3D View ----------------
    ax_3d = fig.add_subplot(2, 2, 1, projection="3d")
    stride = max(1, x_mesh.shape[0] // 10)
    
    if show_vectors:
        xs = x_mesh[::stride, ::stride, ::stride].flatten()
        ys = y_mesh[::stride, ::stride, ::stride].flatten()
        zs = z_mesh[::stride, ::stride, ::stride].flatten()
        u = Ex[::stride, ::stride, ::stride].flatten()
        v = Ey[::stride, ::stride, ::stride].flatten()
        w = Ez[::stride, ::stride, ::stride].flatten()
        mags = np.sqrt(u**2 + v**2 + w**2)
        safe_mags = np.where(mags == 0, 1.0, mags)
        
        cmap = plt.cm.plasma
        arrow_colors = cmap(norm(np.clip(mags, vmin, vmax)))
        grid_span = max(x_mesh.max() - x_mesh.min(), 1.0)
        arrow_len = grid_span / (x_mesh.shape[0] / stride) * 0.65
        
        ax_3d.quiver(
            xs, ys, zs, u / safe_mags, v / safe_mags, w / safe_mags,
            length=arrow_len, normalize=False, colors=arrow_colors,
            arrow_length_ratio=0.35, linewidth=0.8, alpha=0.8
        )
    _draw_charges_3d(ax_3d, charges)
    ax_3d.set_xlabel(r"$x\;(\mathrm{m})$", labelpad=6)
    ax_3d.set_ylabel(r"$y\;(\mathrm{m})$", labelpad=6)
    ax_3d.set_zlabel(r"$z\;(\mathrm{m})$", labelpad=6)
    ax_3d.set_title("(a) 3D Overview $\\mathbf{E}(\\mathbf{r})$", pad=10)

    # ---------------- Panel Helper ----------------
    def render_slice_panel(sub_pos: int, plane: str, u_grid, v_grid, eu, ev, ew, u_idx, v_idx, u_lbl, v_lbl, cut_lbl, panel_tag):
        ax = fig.add_subplot(2, 2, sub_pos)
        mag_slice = np.sqrt(eu**2 + ev**2 + ew**2)
        cf = ax.contourf(u_grid, v_grid, mag_slice, levels=levels, cmap="plasma", norm=norm, extend="both")
        
        if show_vectors:
            step_u = max(1, u_grid.shape[0] // 16)
            step_v = max(1, v_grid.shape[1] // 16)
            qu = u_grid[::step_u, ::step_v]
            qv = v_grid[::step_u, ::step_v]
            q_eu = eu[::step_u, ::step_v]
            q_ev = ev[::step_u, ::step_v]
            q_mag = np.hypot(q_eu, q_ev)
            safe_q = np.where(q_mag == 0, 1.0, q_mag)
            ax.quiver(
                qu, qv, q_eu / safe_q, q_ev / safe_q,
                color="white", scale=26, width=0.0035, headwidth=4, headlength=4.5, alpha=0.85
            )
            
        _draw_charges_2d(ax, charges, u_idx, v_idx, u_lbl, v_lbl)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(u_lbl)
        ax.set_ylabel(v_lbl)
        ax.set_title(f"{panel_tag} {plane.upper()} Plane ({cut_lbl})")
        return cf

    # ---------------- Panel (b): XY Plane (z = slice_coords[2]) ----------------
    z_coords = z_mesh[0, 0, :]
    iz = int(np.argmin(np.abs(z_coords - slice_coords[2])))
    cf_xy = render_slice_panel(
        2, "xy", x_mesh[:, :, iz], y_mesh[:, :, iz], Ex[:, :, iz], Ey[:, :, iz], Ez[:, :, iz],
        0, 1, r"$x\;(\mathrm{m})$", r"$y\;(\mathrm{m})$", f"$z={z_coords[iz]:.2f}\\,\\mathrm{{m}}$", "(b)"
    )

    # ---------------- Panel (c): XZ Plane (y = slice_coords[1]) ----------------
    y_coords = y_mesh[0, :, 0]
    iy = int(np.argmin(np.abs(y_coords - slice_coords[1])))
    render_slice_panel(
        3, "xz", x_mesh[:, iy, :], z_mesh[:, iy, :], Ex[:, iy, :], Ez[:, iy, :], Ey[:, iy, :],
        0, 2, r"$x\;(\mathrm{m})$", r"$z\;(\mathrm{m})$", f"$y={y_coords[iy]:.2f}\\,\\mathrm{{m}}$", "(c)"
    )

    # ---------------- Panel (d): YZ Plane (x = slice_coords[0]) ----------------
    x_coords = x_mesh[:, 0, 0]
    ix = int(np.argmin(np.abs(x_coords - slice_coords[0])))
    render_slice_panel(
        4, "yz", y_mesh[ix, :, :], z_mesh[ix, :, :], Ey[ix, :, :], Ez[ix, :, :], Ex[ix, :, :],
        1, 2, r"$y\;(\mathrm{m})$", r"$z\;(\mathrm{m})$", f"$x={x_coords[ix]:.2f}\\,\\mathrm{{m}}$", "(d)"
    )

    # Add shared colorbar
    fig.subplots_adjust(right=0.90, hspace=0.28, wspace=0.22)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.018, 0.70])
    cbar = fig.colorbar(cf_xy, cax=cbar_ax)
    cbar.set_label(r"Electric Field Magnitude $|\mathbf{E}|\;(\mathrm{V/m})$", rotation=270, labelpad=18)
    
    fig.suptitle(title, fontsize=15, y=0.98)
    return fig
