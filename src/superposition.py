# -*- coding: utf-8 -*-
# By: Mohammad H. Tahersima
# At: 2026-09-04
# ALL RIGHTS RESERVED
"""Compute and visualize electric fields and potentials for arbitrary point charges.

This module utilizes a vectorized brute-force approach to calculate the electrostatic 
potential and field vectors over a 2D mesh grid. It isolates physical constants, 
configuration parameters, and visualization logic for maximum testability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

# Configure module-level logger to replace standard print statements
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ElectrostaticConfig:
    """Configuration constraints and physical constants for the simulation."""
    
    num_charges: int = 15
    spacing_um: float = 0.1
    bound_min_um: float = -10.0
    bound_max_um: float = 10.0
    
    # Physical constants
    coulomb_k: float = 8.99e9      # N·m²/C²
    epsilon_0: float = 8.85e-12    # C²/N·m²
    um_to_m: float = 1e-6          # Conversion factor


def generate_mesh(config: ElectrostaticConfig) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate a 2D spatial mesh grid bounded by the configuration constraints.
    
    Returns:
        A tuple of X and Y coordinate grids in micrometers.
    """
    coords = np.arange(config.bound_min_um, config.bound_max_um + config.spacing_um, config.spacing_um)
    x_mesh, y_mesh = np.meshgrid(coords, coords)
    return x_mesh, y_mesh


def place_random_charges(config: ElectrostaticConfig) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Distribute charges uniformly across the simulation bounds.
    
    Returns:
        A tuple containing the (N, 2) position array and the (N,) charge magnitude array.
    """
    positions = np.random.uniform(
        low=config.bound_min_um,
        high=config.bound_max_um,
        size=(config.num_charges, 2)
    )
    # Assign charges between -5 and 5 μC
    charges_uc = np.random.uniform(-5.0, 5.0, config.num_charges)
    return positions, charges_uc


def calculate_electric_fields(
    x_um: npt.NDArray[np.float64], 
    y_um: npt.NDArray[np.float64], 
    positions_um: npt.NDArray[np.float64], 
    charges_uc: npt.NDArray[np.float64],
    config: ElectrostaticConfig
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Vectorized calculation of electric potential and field components.
    
    Business Intent: Evaluating fields via an explicit Python loop over thousands of 
    grid points is extremely slow. By broadcasting the arrays to (N, M, K) dimensions,
    we push the summation entirely into optimized C-level NumPy routines.
    """
    try:
        # Convert inputs to standard SI units (Meters and Coulombs)
        x_m = x_um * config.um_to_m
        y_m = y_um * config.um_to_m
        pos_x_m = positions_um[:, 0] * config.um_to_m
        pos_y_m = positions_um[:, 1] * config.um_to_m
        q_c = charges_uc * 1e-6
        
        # Broadcast grids against charge positions: result shape is (Grid_Y, Grid_X, Num_Charges)
        dx_m = x_m[..., np.newaxis] - pos_x_m
        dy_m = y_m[..., np.newaxis] - pos_y_m
        r_m = np.hypot(dx_m, dy_m)
        
        # Look Before You Leap (LBYL): Prevent division by zero at singularity points
        r_m = np.where(r_m < 1e-15, 1e-15, r_m)
        
        # Compute superpositions by summing along the third axis (charges)
        v_total = np.sum(config.coulomb_k * q_c / r_m, axis=-1)
        
        ex_m = np.sum(config.coulomb_k * q_c * dx_m / (r_m**3), axis=-1)
        ey_m = np.sum(config.coulomb_k * q_c * dy_m / (r_m**3), axis=-1)
        
        # Convert electric field back to V/μm for domain-specific visualization
        ex_um = ex_m * config.um_to_m
        ey_um = ey_m * config.um_to_m
        
        return v_total, ex_um, ey_um

    except Exception as e:
        logger.error(f"Failed to calculate electric fields: {e}")
        raise


def render_simulations(
    x: npt.NDArray[np.float64], 
    y: npt.NDArray[np.float64], 
    v: npt.NDArray[np.float64], 
    ex: npt.NDArray[np.float64], 
    ey: npt.NDArray[np.float64], 
    positions: npt.NDArray[np.float64],
    config: ElectrostaticConfig
) -> None:
    """Render combined visualizations for potential heatmaps and field streamlines."""
    bounds = [config.bound_min_um, config.bound_max_um, config.bound_min_um, config.bound_max_um]
    
    # Generate combined field and potential figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    
    # Potential background
    im = ax.imshow(
        v, extent=bounds, origin='lower', cmap='viridis', 
        alpha=0.8, aspect='equal', interpolation='bilinear'
    )
    plt.colorbar(im, ax=ax, label='Electric Potential (V)')
    
    # Vector Field Streamlines (Sub-sampled for rendering performance)
    stride = max(1, int(1.0 / config.spacing_um))
    e_mag = np.hypot(ex, ey)
    
    ax.streamplot(
        x[::stride, ::stride], y[::stride, ::stride], 
        ex[::stride, ::stride], ey[::stride, ::stride],
        color='white', density=1.5, linewidth=0.8, arrowsize=1.0
    )
    
    # Overlay active charges
    ax.scatter(positions[:, 0], positions[:, 1], c='red', s=60, edgecolors='black', zorder=5)
    
    ax.set(
        xlabel='X position (μm)', 
        ylabel='Y position (μm)', 
        title='Electrostatic Potential & Field Streamlines'
    )
    
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Execute the core pipeline for the electrostatic simulation."""
    config = ElectrostaticConfig()
    
    logger.info("Initializing spatial grid...")
    x, y = generate_mesh(config)
    
    logger.info(f"Placing {config.num_charges} random point charges...")
    charge_positions, charges = place_random_charges(config)
    
    logger.info("Computing electrostatic fields via vectorized superposition...")
    v, ex, ey = calculate_electric_fields(x, y, charge_positions, charges, config)
    
    logger.info(
        f"Results -> V range: [{v.min():.2e}, {v.max():.2e}] V | "
        f"E_mag max: {np.hypot(ex, ey).max():.2e} V/μm"
    )
    
    logger.info("Rendering visual outputs...")
    try:
        render_simulations(x, y, v, ex, ey, charge_positions, config)
    except Exception as e:
        logger.error(f"Visualization failed: {e}")


if __name__ == "__main__":
    main()
