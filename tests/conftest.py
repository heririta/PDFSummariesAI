"""
Pytest Configuration and Fixtures
Shared test configuration and fixtures for all test modules
"""

import pytest
import tempfile
import os
from pathlib import Path
import sys

# Add src to Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_pdf_path():
    """Path to a sample PDF file for testing"""
    # TODO: Create or provide a sample PDF for testing
    return os.path.join(os.path.dirname(__file__), "fixtures", "sample.pdf")


@pytest.fixture
def sample_text():
    """Sample text for testing"""
    return """
    This is a sample text document for testing the PDF summarizer application.
    It contains multiple sentences and paragraphs to simulate real content.
    The text should be long enough to test the chunking functionality.
    It includes various types of information that might be found in a real document.
    """


@pytest.fixture
def mock_groq_api_key():
    """Mock Groq API key for testing"""
    return "test_groq_api_key_12345"