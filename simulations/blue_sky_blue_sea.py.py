"""Rayleigh scattering physical formulations.

This module provides the pure mathematical logic to calculate 
Rayleigh scattering intensity. It is decoupled from any UI framework 
to ensure testability and reusability across simulations.
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

def calculate_rayleigh_intensity(wavelength_nm: float, reference_nm: float = 700.0) -> float:
    """Calculate the normalized Rayleigh scattering intensity.
    
    Args:
        wavelength_nm: Target wavelength in nanometers.
        reference_nm: Baseline wavelength for relative normalization.
        
    Returns:
        Normalized scattering intensity.
        
    Raises:
        ValueError: If wavelengths are less than or equal to zero.
    """
    if wavelength_nm <= 0 or reference_nm <= 0:
        logger.error("Wavelengths must be strictly positive numbers.")
        raise ValueError("Wavelengths must be strictly positive numbers.")
        
    try:
        base_intensity = 1.0 / (wavelength_nm ** 4)
        ref_intensity = 1.0 / (reference_nm ** 4)
        return float(base_intensity / ref_intensity)
    except Exception as e:
        logger.error(f"Math calculation failed: {e}")
        raise
