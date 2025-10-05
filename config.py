"""
Configuration settings for PDF Summarizer application

This file provides backward compatibility for the new configuration system.
The main configuration is now handled by src.utils.config module.
"""

import warnings
from src.utils.config import get_config, get_groq_api_key, get_database_url, get_max_file_size, get_chunk_size, get_chunk_overlap

# Show deprecation warning
warnings.warn(
    "Using root config.py is deprecated. Please use src.utils.config module instead.",
    DeprecationWarning,
    stacklevel=2
)

# Get configuration instance
config = get_config()

# Backward compatibility exports
GROQ_API_KEY = get_groq_api_key()
GROQ_MODEL = config.groq.model
DATABASE_URL = get_database_url()
MAX_FILE_SIZE = get_max_file_size()
ALLOWED_EXTENSIONS = config.file_upload.allowed_extensions
CHUNK_SIZE = get_chunk_size()
CHUNK_OVERLAP = get_chunk_overlap()
MAX_TOKENS = config.text_processing.max_tokens
PAGE_TITLE = config.streamlit.page_title
PAGE_ICON = config.streamlit.page_icon
LOG_LEVEL = config.app.log_level
LOG_FILE = f"{config.directories.logs_dir}/app.log"

# Export configuration for direct access
__all__ = [
    'config',
    'GROQ_API_KEY',
    'GROQ_MODEL',
    'DATABASE_URL',
    'MAX_FILE_SIZE',
    'ALLOWED_EXTENSIONS',
    'CHUNK_SIZE',
    'CHUNK_OVERLAP',
    'MAX_TOKENS',
    'PAGE_TITLE',
    'PAGE_ICON',
    'LOG_LEVEL',
    'LOG_FILE',
    'get_config',
    'get_groq_api_key',
    'get_database_url',
    'get_max_file_size',
    'get_chunk_size',
    'get_chunk_overlap'
]