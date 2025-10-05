"""
PDF Summarizer AI - Main Application Entry Point

This is the main entry point for the PDF Summarizer AI application.
Run this file with: streamlit run app.py
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import application modules
from ui.app import PDFSummarizerApp
from utils.config import get_config
from utils.logger import setup_logger

def main():
    """Main application entry point"""
    try:
        # Setup logging
        logger = setup_logger()
        logger.info("Starting PDF Summarizer AI Application")

        # Get configuration
        config = get_config()

        # Initialize and run the Streamlit app
        app = PDFSummarizerApp(config)
        app.run()

    except Exception as e:
        st.error(f"Failed to start application: {str(e)}")
        if 'logger' in locals():
            logger.error(f"Application startup error: {e}")
        else:
            print(f"Application startup error: {e}")

if __name__ == "__main__":
    main()