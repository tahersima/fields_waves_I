# Author: Mohammad H. Tahersima
# Date: Julu 2026
# ALL RIGHTS RESERVED 

# ==========================================
# File: app.py (The Main Application Router)
# ==========================================

import logging
from pathlib import Path
import streamlit as st

# Configure root logger for visibility into runtime execution
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def enforce_page_existence(filepath: str) -> None:
    """Ensure the target page file exists on the filesystem to prevent Streamlit routing failures.
    
    Args:
        filepath: The relative path to the expected Streamlit page file.
    """
    path = Path(filepath)
    if not path.is_file():
        logger.warning(f"Required page file not found, creating placeholder: {filepath}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f'import streamlit as st\nst.title("Placeholder for {path.stem}")\n'
        )


def main() -> None:
    """Initialize the Streamlit multipage application routing for simulations only."""
    try:
        st.set_page_config(
            page_title="Electromagnetics: Fields & Waves", 
            page_icon="🌊", 
            layout="wide"
        )
        
        # Look before you leap (LBYL): Verify filesystem state before passing to the router
        page_paths = [
            "pages/1_simulate_chapter_1.py",
            "pages/2_simulate_chapter_2.py"
        ]
        for path in page_paths:
            enforce_page_existence(path)

        # Construct the navigation menu focused exclusively on available simulations
        nav = st.navigation({
            "Chapter 1: Blue of the Sky, Blue of the Sea": [
                st.Page("pages/1_simulate_chapter_1.py", title="Rayleigh Scattering Simulation", icon="🎛️"),
            ],
            "Chapter 2: Electrostatics & Vectors": [
                st.Page("pages/2_simulate_chapter_2.py", title="Electrostatic Field Simulation", icon="🎛️"),
            ]
        })
        nav.run()
        
    except AttributeError as e:
        logger.error(f"Streamlit API version incompatibility: {e}")
        st.error(
            "Routing failed. The `st.navigation` feature requires Streamlit version 1.36 or newer. "
            "Please upgrade your environment: `pip install --upgrade streamlit`"
        )
    except Exception as e:
        logger.error(f"Failed to initialize application routing: {e}")
        st.error(f"A critical routing error occurred: {e}")


if __name__ == "__main__":
    main()
