# -*- coding: utf-8 -*-
# By: Mohammad H. Tahersima
# At: 2026-09-04
# ALL RIGHTS RESERVED
"""Coulomb's Law Simulation Engine: Electric Force & Field Distribution.

Course: ECE 3315 - Fields and Waves I
Audience: Undergraduate Engineering Students
Environment: Python 3 / Spyder (Anaconda)

This simulation computes:
  1. Spatial electric field E(r) in 3D using Coulomb's law:
     E(r) = 1 / (4 * pi * epsilon_0) * sum_i [ q_i * (r - r_i) / |r - r_i|^3 ]
  2. Total electrostatic force F acting on an optional test charge:
     F = q_test * E(r_test)
  3. Publication-ready visualizations:
     - Separate 3D spatial field distribution
     - Separate 2D planar cross-sections (XY, XZ, YZ)
     - Combined 2x2 multi-panel publication summary
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import scipy.constants as const

# Ensure the repository root is on sys.path so 'src.plotting' imports seamlessly in Spyder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
from src.plotting import (
    apply_publication_style,
    plot_field_3d,
    plot_field_2d_slice,
    plot_field_2x2_summary
)


# ==============================================================================
# USER CONFIGURATION (Edit these parameters to explore different problems!)
# ==============================================================================

# 1. Coordinate System Selection for Input Coordinates:
#    Options: 'cartesian', 'cylindrical', 'spherical'
#    - 'cartesian':   (x, y, z) in meters
#    - 'cylindrical': (rho in meters, phi in degrees, z in meters)
#    - 'spherical':   (r in meters, theta [0, 180 deg], phi [0, 360 deg])
COORD_SYSTEM = "cartesian"

# 2. Source Charges: List of positions in COORD_SYSTEM and charge values (Coulombs)
#    Default example: Electric dipole (+2 nC at (0, 0, 1) and -2 nC at (0, 0, -1))
CHARGES_INPUT = [
    {"pos": (0.0, 0.0,  1.0), "q": +2.0e-9},   # +2 nanoCoulombs
    {"pos": (0.0, 0.0, -1.0), "q": -2.0e-9},   # -2 nanoCoulombs
    {"pos": (0.5, 2.0, 0.0), "q": -2.0e-9},   # -2 nanoCoulombs 
]

# 3. Optional Test Charge for Force Evaluation: F = q_test * E(r_test)
#    Set to None if not evaluating force on a specific charge.
TEST_CHARGE_INPUT = {
    "pos": (0.0, 1.0, 0.0),   # Position in COORD_SYSTEM
    "q": +1.0e-9              # +1 nC test charge
}

# 4. Computational Grid Resolution and Bounds (in meters):
GRID_BOUNDS = (-3.0, 3.0)     # Bounds [min, max] along X, Y, and Z
GRID_RESOLUTION = 100          # Number of grid points per axis (e.g., 30-50 for quick rendering)

# 5. Visualization Flags:
SHOW_VECTORS = True           # True = overlay vector quiver arrows (magnitude + direction)
SLICE_PLANES = (0.0, 0.0, 0.0)# Planar cuts (x_cut, y_cut, z_cut) for 2D cross-sections
SHOW_SEPARATE_PLOTS = True    # True = display individual 3D, XY, XZ, YZ figure windows
SHOW_2X2_SUMMARY = True       # True = display unified 2x2 multi-panel publication figure


# ==============================================================================
# Coordinate Transformation Utilities
# ==============================================================================
def to_cartesian(coords: tuple[float, float, float], system: str) -> np.ndarray:
    """Convert coordinates from Cartesian, Cylindrical, or Spherical to Cartesian (x, y, z).
    
    Args:
        coords: Tuple of 3 numerical values.
        system: 'cartesian', 'cylindrical', or 'spherical'.
        
    Returns:
        NumPy array [x, y, z] in meters.
    """
    sys_lower = system.lower().strip()
    
    if sys_lower == "cartesian":
        return np.array(coords, dtype=np.float64)
    
    elif sys_lower == "cylindrical":
        # coords = (rho, phi_deg, z)
        rho, phi_deg, z = coords
        phi_rad = np.radians(phi_deg)
        x = rho * np.cos(phi_rad)
        y = rho * np.sin(phi_rad)
        return np.array([x, y, z], dtype=np.float64)
        
    elif sys_lower == "spherical":
        # coords = (r, theta_deg, phi_deg)
        r, theta_deg, phi_deg = coords
        theta_rad = np.radians(theta_deg)
        phi_rad = np.radians(phi_deg)
        x = r * np.sin(theta_rad) * np.cos(phi_rad)
        y = r * np.sin(theta_rad) * np.sin(phi_rad)
        z = r * np.cos(theta_rad)
        return np.array([x, y, z], dtype=np.float64)
        
    else:
        raise ValueError(f"Unknown coordinate system: '{system}'. Use 'cartesian', 'cylindrical', or 'spherical'.")


# ==============================================================================
# Physical Constants Tracking (via scipy.constants)
# ==============================================================================
def display_physical_constants() -> float:
    """Print the fundamental physical constants with professional units and uncertainties."""
    eps_val, eps_unit, eps_unc = const.physical_constants["vacuum electric permittivity"]
    e_val, e_unit, e_unc = const.physical_constants["elementary charge"]
    
    k_coulomb = 1.0 / (4.0 * const.pi * eps_val)
    
    print("=" * 70)
    print("FUNDAMENTAL PHYSICAL CONSTANTS (CODATA via scipy.constants):")
    print(f"  - Permittivity of Free Space (eps_0): {eps_val:.12e} {eps_unit}  (std unc: {eps_unc})")
    print(f"  - Elementary Charge (e):             {e_val:.12e} {e_unit}  (std unc: {e_unc})")
    print(f"  - Coulomb Constant (k_e = 1/4*pi*eps_0): {k_coulomb:.6e} N*m^2/C^2")
    print("=" * 70)
    return k_coulomb


# ==============================================================================
# Physics Computation: Coulomb Electric Field & Force
# ==============================================================================
def compute_electric_field_at_point(
    r_obs: np.ndarray, 
    cartesian_charges: list[dict], 
    k_coulomb: float
) -> np.ndarray:
    """Compute the electric field vector E = (Ex, Ey, Ez) at an arbitrary observation point r_obs."""
    E_total = np.zeros(3, dtype=np.float64)
    for chg in cartesian_charges:
        r_src = chg["pos"]
        q = chg["q"]
        delta_r = r_obs - r_src
        dist = np.linalg.norm(delta_r)
        if dist < 1e-12:
            # Observation point is directly on the charge (singularity)
            continue
        E_total += k_coulomb * q * delta_r / (dist**3)
    return E_total


def compute_3d_grid_fields(
    x_mesh: np.ndarray,
    y_mesh: np.ndarray,
    z_mesh: np.ndarray,
    cartesian_charges: list[dict],
    k_coulomb: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized calculation of 3D electric field components across the grid."""
    Ex = np.zeros_like(x_mesh, dtype=np.float64)
    Ey = np.zeros_like(y_mesh, dtype=np.float64)
    Ez = np.zeros_like(z_mesh, dtype=np.float64)

    for chg in cartesian_charges:
        x0, y0, z0 = chg["pos"]
        q = chg["q"]

        dx = x_mesh - x0
        dy = y_mesh - y0
        dz = z_mesh - z0

        dist_sq = dx**2 + dy**2 + dz**2
        # Soften singularity right at the charge center to avoid dividing by zero
        dist_sq_safe = np.where(dist_sq < 1e-10, 1e-10, dist_sq)
        dist_cube = dist_sq_safe * np.sqrt(dist_sq_safe)

        factor = k_coulomb * q / dist_cube
        Ex += factor * dx
        Ey += factor * dy
        Ez += factor * dz

    return Ex, Ey, Ez


# ==============================================================================
# Main Simulation Execution (Spyder Entry Point)
# ==============================================================================
def main() -> None:
    apply_publication_style()
    k_e = display_physical_constants()

    # 1. Convert all charges into Cartesian coordinates
    cartesian_charges = []
    print(f"\nSOURCE CHARGES (Input system: {COORD_SYSTEM.upper()}):")
    for idx, c in enumerate(CHARGES_INPUT, start=1):
        pos_cart = to_cartesian(c["pos"], COORD_SYSTEM)
        cartesian_charges.append({"pos": pos_cart, "q": c["q"]})
        print(f"  Charge q_{idx}: {c['q']:+.2e} C | Input: {c['pos']} -> Cartesian: ({pos_cart[0]:.2f}, {pos_cart[1]:.2f}, {pos_cart[2]:.2f}) m")

    # 2. Evaluate Force on Test Charge (if specified)
    if TEST_CHARGE_INPUT is not None:
        pos_test_cart = to_cartesian(TEST_CHARGE_INPUT["pos"], COORD_SYSTEM)
        q_test = TEST_CHARGE_INPUT["q"]
        E_at_test = compute_electric_field_at_point(pos_test_cart, cartesian_charges, k_e)
        F_on_test = q_test * E_at_test
        F_mag = np.linalg.norm(F_on_test)
        
        print("\n" + "-" * 70)
        print(f"TEST CHARGE EVALUATION (q_test = {q_test:+.2e} C at r = ({pos_test_cart[0]:.2f}, {pos_test_cart[1]:.2f}, {pos_test_cart[2]:.2f}) m):")
        print(f"  - Electric Field: E = ({E_at_test[0]:+.3e}, {E_at_test[1]:+.3e}, {E_at_test[2]:+.3e}) V/m")
        print(f"  - Field Magnitude: |E| = {np.linalg.norm(E_at_test):.3e} V/m")
        print(f"  - Resultant Force: F = ({F_on_test[0]:+.3e}, {F_on_test[1]:+.3e}, {F_on_test[2]:+.3e}) N")
        print(f"  - Force Magnitude: |F| = {F_mag:.3e} N")
        print("-" * 70)

    # 3. Create 3D Meshgrid
    print(f"\nComputing 3D electric field distribution ({GRID_RESOLUTION}x{GRID_RESOLUTION}x{GRID_RESOLUTION} points)...")
    lin = np.linspace(GRID_BOUNDS[0], GRID_BOUNDS[1], GRID_RESOLUTION)
    x_mesh, y_mesh, z_mesh = np.meshgrid(lin, lin, lin, indexing="ij")

    # 4. Compute Vector Field
    Ex, Ey, Ez = compute_3d_grid_fields(x_mesh, y_mesh, z_mesh, cartesian_charges, k_e)
    E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    print(f"Computation complete! |E| range: [{E_mag.min():.2e}, {E_mag.max():.2e}] V/m")

    # 5. Render Visualizations
    print("\nGenerating scientific-grade plots...")

    if SHOW_SEPARATE_PLOTS:
        # Separate Figure 1: 3D Global Perspective
        plot_field_3d(x_mesh, y_mesh, z_mesh, Ex, Ey, Ez, cartesian_charges, show_vectors=SHOW_VECTORS)

        # Separate Figure 2: XY Plane Cross-Section
        plot_field_2d_slice(x_mesh, y_mesh, z_mesh, Ex, Ey, Ez, cartesian_charges, plane="xy", slice_coord=SLICE_PLANES[2], show_vectors=SHOW_VECTORS)

        # Separate Figure 3: XZ Plane Cross-Section
        plot_field_2d_slice(x_mesh, y_mesh, z_mesh, Ex, Ey, Ez, cartesian_charges, plane="xz", slice_coord=SLICE_PLANES[1], show_vectors=SHOW_VECTORS)

        # Separate Figure 4: YZ Plane Cross-Section
        plot_field_2d_slice(x_mesh, y_mesh, z_mesh, Ex, Ey, Ez, cartesian_charges, plane="yz", slice_coord=SLICE_PLANES[0], show_vectors=SHOW_VECTORS)

    if SHOW_2X2_SUMMARY:
        # Separate Figure 5: Unified 2x2 Multi-Panel Publication Summary
        plot_field_2x2_summary(x_mesh, y_mesh, z_mesh, Ex, Ey, Ez, cartesian_charges, show_vectors=SHOW_VECTORS, slice_coords=SLICE_PLANES)

    print("Displaying plots. (In Spyder, plots appear in the Plots pane).")
    plt.show()


if __name__ == "__main__":
    main()
