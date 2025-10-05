#!/usr/bin/env python3
"""
PDF Summarizer Application
Main entry point for the Streamlit application
"""

import streamlit as st
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ui.app import run_app

if __name__ == "__main__":
    run_app()