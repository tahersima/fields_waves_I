# -*- coding: utf-8 -*-
# By: Mohammad H. Tahersima
# At: 2026-09-04
# ALL RIGHTS RESERVED 
"""Mathematical engine for calculating 3D scalar potentials and gradients.

This module is strictly decoupled from any visualization framework, allowing
it to be imported by testing suites, Jupyter notebooks, or Streamlit apps.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import numpy.typing as npt

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
    """Create a 3D spatial meshgrid."""
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
    """Calculate the scalar potential field and its numerical spatial gradients."""
    try:
        field = potential_func(x_mesh, y_mesh, z_mesh)
        dx, dy, dz = np.gradient(field, coords, coords, coords, edge_order=2)
        return field, dx, dy, dz
    except Exception as e:
        logger.error(f"Failed to compute field gradients: {e}")
        raise


def sample_electric_potential(
    x: npt.NDArray[np.float64], 
    y: npt.NDArray[np.float64], 
    z: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Sample continuous scalar potential function."""
    return (x**3) * y * z + 0.2 * (z**2)
