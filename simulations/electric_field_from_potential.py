# Mohammad H. Tahersima / All Rights Reserved 
"""Compute and plot electric potential distributions and corresponding fields.

This module calculates the 3D scalar potential field and derives the electric
field vectors using numerical gradients. It demonstrates Pythonic array manipulations,
continuous grid interpolation, and decoupled visualization logic.
"""


"""Streamlit micro-app for visualizing electric potential and fields."""
from __future__ import annotations

import logging
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import streamlit as st
from scipy.interpolate import RegularGridInterpolator

from src.field_math import GridConfig, create_grid, calculate_field_and_gradients, sample_electric_potential

logger = logging.getLogger(__name__)


def plot_gradient_vectors(
    x_mesh: npt.NDArray[np.float64],
    y_mesh: npt.NDArray[np.float64],
    field: npt.NDArray[np.float64],
    z_slice_index: int
) -> None:
    """Render the 2D cross-section of the potential field."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    contour = ax.contourf(
        x_mesh[:, :, z_slice_index], 
        y_mesh[:, :, z_slice_index], 
        field[:, :, z_slice_index], 
        20, 
        cmap="viridis"
    )
    plt.colorbar(contour, ax=ax, label="Potential Field Strength (V)")
    ax.set(xlabel="X position", ylabel="Y position", title="2D Potential Cross-Section")
    ax.set_aspect("equal", adjustable="box")
    st.pyplot(fig)
    plt.close(fig)


def plot_electric_field(
    dx: npt.NDArray[np.float64],
    dy: npt.NDArray[np.float64],
    coords: npt.NDArray[np.float64],
    z_slice_index: int,
    config: GridConfig,
    num_points: int = 20
) -> None:
    """Render the electric field vector quiver plot."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    coords_e = np.linspace(config.start + 0.5, config.stop - 0.5, num_points)
    x_e, y_e = np.meshgrid(coords_e, coords_e, indexing="ij")

    interp_dx = RegularGridInterpolator((coords, coords, coords), dx)
    interp_dy = RegularGridInterpolator((coords, coords, coords), dy)

    z_val = coords[z_slice_index]
    query_points = np.stack([
        x_e.ravel(), 
        y_e.ravel(), 
        np.full_like(x_e.ravel(), z_val)
    ], axis=-1)

    ex = -interp_dx(query_points).reshape(x_e.shape)
    ey = -interp_dy(query_points).reshape(y_e.shape)

    ax.quiver(
        x_e, y_e, ex, ey, 
        color="blue", scale=25, width=0.004, 
        label=r"Electric Field ($E = -\nabla\varphi$)"
    )
    ax.legend(loc="upper right")
    ax.set(xlabel="X position", ylabel="Y position", title="Electric Field Vector Field")
    ax.set_aspect("equal", adjustable="box")
    st.pyplot(fig)
    plt.close(fig)


def main() -> None:
    st.title("Electric Potential & Gradient Visualization")
    st.write("Explore the relationship between scalar potential $\\varphi$ and the electric field $E = -\\nabla\\varphi$.")
    
    config = GridConfig(size=100)
    
    @st.cache_data
    def load_field_data() -> tuple:
        x, y, z, coords = create_grid(config)
        field, dx, dy, dz = calculate_field_and_gradients(sample_electric_potential, x, y, z, coords)
        return x, y, z, coords, field, dx, dy, dz

    with st.spinner("Computing 3D field gradients..."):
        x, y, z, coords, field, dx, dy, dz = load_field_data()

    st.sidebar.header("Simulation Controls")
    z_slice = st.sidebar.slider("Z-Axis Cross-Section", 0, config.size - 1, config.size // 2)
    
    col1, col2 = st.columns(2)
    with col1:
        plot_gradient_vectors(x, y, field, z_slice)
    with col2:
        plot_electric_field(dx, dy, coords, z_slice, config)


if __name__ == "__main__":
    main()
