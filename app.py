# Author: Mohammad H. Tahersima
# Date: Julu 2026
# ALL RIGHTS RESERVED 

# ==========================================
# File: app.py (The Main Application Router)
# ==========================================
import logging
import streamlit as st

# Configure root logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main() -> None:
    """Initialize the Streamlit multipage application routing."""
    try:
        st.set_page_config(
            page_title="Electromagnetics: Fields & Waves", 
            page_icon="🌊", 
            layout="wide"
        )
        
        # Construct the navigation menu for the book 
        nav = st.navigation({
            "Chapter 1: Blue of the Sky, Blue of the Sea": [
                st.Page("pages/1_read_chapter_1.py", title="Reading Mode", icon="📖"),
                st.Page("pages/2_simulate_chapter_1.py", title="Simulation Mode", icon="🎛️"),
            ],
            "Chapter 2: Maxwell's Equations": [
                st.Page("pages/3_read_chapter_2.py", title="Reading Mode", icon="📖"),
                st.Page("pages/4_simulate_chapter_2.py", title="Simulation Mode", icon="🎛️"),
            ]
        })
        nav.run()
        
    except Exception as e:
        logger.error(f"Failed to initialize application routing: {e}")
        st.error("A critical routing error occurred. Please check the logs.")

if __name__ == "__main__":
    main()
