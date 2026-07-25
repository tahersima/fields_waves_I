"""Rayleigh scattering intensity simulation for electromagnetic waves.

This module provides an interactive Streamlit application to calculate and
visualize how the scattering intensity of light in the atmosphere varies with
its wavelength, adhering to the physical principle that scattering is
inversely proportional to the fourth power of the wavelength (I proportional
to 1 / lambda^4). It serves as a companion digital artifact for an
electromagnetic fields and waves textbook.
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
    """
    if wavelength_nm <= 0 or reference_nm <= 0:
        logger.error("Wavelengths must be strictly positive numbers.")
        raise ValueError("Invalid wavelength provided.")
        
    try:
        base_intensity = 1.0 / (wavelength_nm ** 4)
        ref_intensity = 1.0 / (reference_nm ** 4)
        return float(base_intensity / ref_intensity)
    except Exception as e:
        logger.error(f"Math calculation failed: {e}")
        raise
