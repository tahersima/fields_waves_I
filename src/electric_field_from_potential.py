# Mohammad H. Tahersima / All Rights Reserved 
"""Compute and plot electric potential distributions and corresponding fields.

This module calculates the 3D scalar potential field and derives the electric
field vectors using numerical gradients. It demonstrates Pythonic array manipulations,
continuous grid interpolation, and decoupled visualization logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator

# Configure logging for robustness and runtime observability
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GridConfig:
    """Configuration constraints for the 3D spatial grid."""
    size: int = 100
    start: float = -5.0
    stop: float = 5.0

    def __post_init__(self) -> None:
        """Validate grid boundaries at initialization."""
        if self.start >= self.stop:
            raise ValueError(f"Invalid bounds: start ({self.start}) must be less than stop ({self.stop}).")


def create_grid(
    config: GridConfig
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Create a 3D spatial meshgrid.

    Business Intent: Establishes the discrete coordinate system necessary for
    numerical evaluation of the continuous electromagnetic potential fields.

    Args:
        config: The GridConfig object containing resolution and spatial boundaries.

    Returns:
        A tuple containing (X, Y, Z) meshgrids and the 1D coordinate array.
    """
    coords = np.linspace(config.start, config.stop, config.size)
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    return x, y, z, coords


def calculate_field_and_gradients(
    potential_func: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    x_mesh: npt.NDArray[np.float64],
    y_mesh: npt.NDArray[np.float64],
    z_mesh: npt.NDArray[np.float64],
    coords: npt.NDArray[np.float64]
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate the scalar potential field and its numerical spatial gradients.

    Args:
        potential_func: The mathematical function defining the scalar potential.
        x_mesh: 3D meshgrid for X coordinates.
        y_mesh: 3D meshgrid for Y coordinates.
        z_mesh: 3D meshgrid for Z coordinates.
        coords: 1D array of the grid coordinate points for step size reference.

    Returns:
        A tuple containing the evaluated field array and its dx, dy, dz gradients.
    """
    try:
        field = potential_func(x_mesh, y_mesh, z_mesh)
        # Compute gradient using second-order accurate central differences
        dx, dy, dz = np.gradient(field, coords, coords, coords, edge_order=2)
        return field, dx, dy, dz
    except Exception as e:
        logger.error(f"Failed to compute field gradients: {e}")
        raise


def plot_gradient_vectors(
    ax: plt.Axes,
    x_mesh: npt.NDArray[np.float64],
    y_mesh: npt.NDArray[np.float64],
    field: npt.NDArray[np.float64],
    dx: npt.NDArray[np.float64],
    dy: npt.NDArray[np.float64],
    points: npt.NDArray[np.float64],
    z_slice_index: int,
    config: GridConfig
) -> None:
    """Plot a 2D cross-section of the potential field with specific gradient vectors."""
    contour = ax.contourf(
        x_mesh[:, :, z_slice_index], 
        y_mesh[:, :, z_slice_index], 
        field[:, :, z_slice_index], 
        20, 
        cmap="viridis"
    )
    plt.colorbar(contour, ax=ax, label="Field Strength")

    # Safely convert continuous points to nearest discrete grid indices
    for point in points:
        xi = int(np.clip((point[0] - config.start) * (config.size - 1) / (config.stop - config.start), 0, config.size - 1))
        yi = int(np.clip((point[1] - config.start) * (config.size - 1) / (config.stop - config.start), 0, config.size - 1))
        
        gx = dx[xi, yi, z_slice_index]
        gy = dy[xi, yi, z_slice_index]
        
        ax.quiver(point[0], point[1], gx, gy, color="red", scale=15, width=0.005, headwidth=5, headlength=7)

    ax.scatter(points[:, 0], points[:, 1], c="white", s=50, edgecolors="black")


def plot_electric_field(
    ax: plt.Axes,
    dx: npt.NDArray[np.float64],
    dy: npt.NDArray[np.float64],
    coords: npt.NDArray[np.float64],
    z_slice_index: int,
    config: GridConfig,
    num_points: int = 20
) -> None:
    """Calculate and plot the electric field (E = -∇φ) using continuous grid interpolation.

    Business Intent: Visualizing the continuous electric field requires downsampling 
    the dense computational grid. Utilizing scipy's RegularGridInterpolator ensures we
    extract mathematically rigorous field vectors at the exact visual coordinates,
    entirely bypassing discrete grid-snapping artifacts.

    Args:
        ax: Matplotlib axes object.
        dx: X-component of the field gradient on the dense grid.
        dy: Y-component of the field gradient on the dense grid.
        coords: The 1D coordinate array mapping the dense grid.
        z_slice_index: The index of the Z plane to slice.
        config: Dense grid spatial configuration.
        num_points: Resolution of the coarse visual grid.
    """
    coords_e = np.linspace(config.start + 0.5, config.stop - 0.5, num_points)
    x_e, y_e = np.meshgrid(coords_e, coords_e, indexing="ij")

    # Establish continuous 3D interpolators from the discrete gradient data
    interp_dx = RegularGridInterpolator((coords, coords, coords), dx)
    interp_dy = RegularGridInterpolator((coords, coords, coords), dy)

    # Construct an N x 3 array of precise continuous query points for the chosen Z-plane
    z_val = coords[z_slice_index]
    query_points = np.stack([
        x_e.ravel(), 
        y_e.ravel(), 
        np.full_like(x_e.ravel(), z_val)
    ], axis=-1)

    # E = -∇φ (Evaluated exactly at the floating-point coordinates)
    ex = -interp_dx(query_points).reshape(x_e.shape)
    ey = -interp_dy(query_points).reshape(y_e.shape)

    ax.quiver(
        x_e, y_e, ex, ey, 
        color="blue", scale=25, width=0.004, headwidth=3, headlength=5, 
        label=r"Electric Field ($E = -\nabla\varphi$)"
    )
    ax.legend(loc="upper right")


def electric_potential(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Sample continuous scalar potential function."""
    return x**3 * y * z + 0.2 * z**2


def main() -> None:
    """Execute main computation and rendering pipeline."""
    config = GridConfig(size=100)
    z_slice_index = config.size // 2
    points = np.array([[-4.0, 3.0, 0.0], [2.0, -1.0, 0.0]]) 

    logger.info("Initializing 3D spatial grid...")
    x, y, z, coords = create_grid(config)

    logger.info("Computing potential field and numerical gradients...")
    field, dx, dy, dz = calculate_field_and_gradients(electric_potential, x, y, z, coords)

    try:
        logger.info("Rendering visualizations...")
        
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        plot_gradient_vectors(ax1, x, y, field, dx, dy, points, z_slice_index, config)
        ax1.set(xlabel="X position", ylabel="Y position", title="2D Cross-Section with Gradient Vectors")
        ax1.set_aspect("equal", adjustable="box")
        ax1.grid(alpha=0.3)
        fig1.tight_layout()
        plt.show()

        fig2, ax2 = plt.subplots(figsize=(4, 4))
        # Now passing the continuous `coords` array for precise interpolation
        plot_electric_field(ax2, dx, dy, coords, z_slice_index, config)
        ax2.set(xlabel="X position", ylabel="Y position", title=r"Electric Field ($E = -\nabla\varphi$)")
        ax2.set_aspect("equal", adjustable="box")
        ax2.grid(alpha=0.3)
        fig2.tight_layout()
        plt.show()

    except Exception as e:
        logger.error(f"Failed to render matplotlib figures: {e}")


if __name__ == "__main__":
    main()
