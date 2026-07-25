# Mohammad H. Tahersima / All Rights Reserved 
"""Chapter 2: Vector Operations and 3D Visualization for Electromagnetic Fields."""

import logging
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

# Configure logging for robustness, replacing standard print statements
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PlotConfig:
    """Configuration constraints for the 3D Cartesian vector visualization."""

    fig_size: tuple[int, int] = (8, 6)
    dpi: int = 150
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    color_e1: str = "blue"
    color_e2: str = "red"
    color_add: str = "green"
    color_cross: str = "purple"


def calculate_vector_angle(
    vec_a: npt.NDArray[np.float64], 
    vec_b: npt.NDArray[np.float64]
) -> float:
    """Calculate the angle in degrees between two N-dimensional vectors.
    
    Business Intent: Understanding the phase and spatial relationship between
    two electromagnetic field vectors is critical for determining wave polarization
    and Poynting vector flow.
    
    Args:
        vec_a: First vector array.
        vec_b: Second vector array.
        
    Returns:
        Angle in degrees between the two vectors.
        
    Raises:
        ValueError: If either vector has a magnitude of zero.
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    # Look before you leap (LBYL): Prevent division by zero errors
    if norm_a == 0 or norm_b == 0:
        logger.error("Zero vector magnitude detected. Angle is undefined.")
        raise ValueError("Cannot compute the angle of a zero vector.")

    unit_a = vec_a / norm_a
    unit_b = vec_b / norm_b

    # Constrain the dot product to valid domain [-1, 1] for arccos 
    # to prevent floating-point nan issues
    dot_product = float(np.clip(np.dot(unit_a, unit_b), -1.0, 1.0))
    angle_rad = np.arccos(dot_product)
    
    return float(np.rad2deg(angle_rad))


def render_vector_visualization(
    e1: npt.NDArray[np.float64], 
    e2: npt.NDArray[np.float64], 
    e_add: npt.NDArray[np.float64], 
    e_cross: npt.NDArray[np.float64], 
    dot_prod: float, 
    angle: float
) -> None:
    """Render a 3D Cartesian plot displaying the fundamental vectors and their resultants.
    
    Args:
        e1: The primary electric field vector.
        e2: The secondary electric field vector.
        e_add: The vector addition resultant.
        e_cross: The cross product resultant.
        dot_prod: The calculated scalar dot product.
        angle: The calculated angle between e1 and e2 in degrees.
    """
    config = PlotConfig()
    
    try:
        fig = plt.figure(figsize=config.fig_size, dpi=config.dpi)
        ax = fig.add_subplot(111, projection='3d')

        # Helper function to keep code DRY when plotting multiple vectors
        def draw
