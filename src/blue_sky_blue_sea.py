"""Rayleigh scattering intensity simulation for electromagnetic waves.

This module provides an interactive Streamlit application to calculate and
visualize how the scattering intensity of light in the atmosphere varies with
its wavelength, adhering to the physical principle that scattering is
inversely proportional to the fourth power of the wavelength (I proportional
to 1 / lambda^4). It serves as a companion digital artifact for an
electromagnetic fields and waves textbook.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Configure logging for robustness
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ScatteringConfig:
    """Configuration constraints for the Rayleigh scattering model."""

    min_wavelength: int = 400
    max_wavelength: int = 700
    default_wavelength: int = 550
    reference_wavelength: float = 700.0


def calculate_relative_intensity(wavelength: float, ref_wavelength: float) -> float:
    """Calculate the normalized Rayleigh scattering intensity based on wavelength.

    Args:
        wavelength: Current wavelength of light in nanometers.
        ref_wavelength: Baseline reference wavelength for normalization.

    Returns:
        The normalized relative scattering intensity.
    """
    try:
        # Rayleigh scattering intensity is inversely proportional to the fourth power of wavelength
        base_intensity = 1.0 / (wavelength ** 4)
        ref_intensity = 1.0 / (ref_wavelength ** 4)
        return float(base_intensity / ref_intensity)
    except ZeroDivisionError as e:
        logger.error("Wavelength cannot be zero.")
        raise ValueError("Wavelength must be greater than zero.") from e


def render_scattering_visualization(relative_intensity: float) -> None:
    """Render a publication-quality Matplotlib bar chart for the Streamlit app.

    Args:
        relative_intensity: The computed relative power of the scattered light.
    """
    try:
        fig, ax = plt.subplots(figsize=(6, 2), dpi=300)
        ax.barh(["Scattering Power"], [relative_intensity], color="dodgerblue")
        ax.set_xlim(0, 15)
        ax.set_xlabel("Normalized Intensity")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        logger.error("Failed to render Matplotlib figure: %s", e)
        st.error("Could not render visualization due to an internal error.")


def main() -> None:
    """Execute the main Streamlit application layout and logic."""
    config = ScatteringConfig()

    st.title("Interactive Rayleigh Scattering Sandbox")
    st.write(
        "Adjust the wavelength of incoming light to see how scattering "
        "intensity changes ($I \\propto 1/\\lambda^4$). "
        "Inspired by Koichi Ota's *Foundations of Electromagnetics*."
    )

    # User input slider adhering to physical visible light boundaries
    wavelength = st.slider(
        "Wavelength (nm)",
        min_value=config.min_wavelength,
        max_value=config.max_wavelength,
        value=config.default_wavelength,
    )

    # Compute intensity utilizing separate, testable logic
    relative_intensity = calculate_relative_intensity(
        float(wavelength), config.reference_wavelength
    )

    st.write(f"Current Wavelength: **{wavelength} nm**")
    st.write(f"Relative Scattering Power: **{relative_intensity:.2f}x**")

    # Render graphical output
    render_scattering_visualization(relative_intensity)


if __name__ == "__main__":
    main()
