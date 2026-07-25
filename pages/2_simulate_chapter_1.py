# ==========================================
# File: pages/2_simulate_chapter_1.py (UI Layer)
# ==========================================
from dataclasses import dataclass
import logging
import matplotlib.pyplot as plt
import streamlit as st

# In a real app, this import connects to your decoupled logic module
# from src.physics import calculate_rayleigh_intensity
from src.physics import calculate_rayleigh_intensity

logger = logging.getLogger(__name__)

@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Constraints for the Rayleigh simulation UI."""
    min_nm: int = 400
    max_nm: int = 700
    default_nm: int = 550

def render_chart(intensity: float) -> None:
    """Render and display the Matplotlib chart."""
    try:
        fig, ax = plt.subplots(figsize=(6, 2), dpi=300)
        ax.barh(["Scattering Power"], [intensity], color="dodgerblue")
        ax.set_xlim(0, 15)
        ax.set_xlabel("Normalized Intensity")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        logger.error(f"Plotting failed: {e}")
        st.error("Failed to render the visualization.")

def render_simulation_page() -> None:
    """Render the interactive simulation UI for Chapter 1."""
    config = SimulationConfig()
    
    st.title("Interactive Rayleigh Scattering")
    wavelength = st.slider(
        "Wavelength (nm)", 
        min_value=config.min_nm, 
        max_value=config.max_nm, 
        value=config.default_nm
    )
    
    intensity = calculate_rayleigh_intensity(float(wavelength))
    st.write(f"Relative Scattering Power: **{intensity:.2f}x**")
    
    render_chart(intensity)

if __name__ == "__main__":
    render_simulation_page()
