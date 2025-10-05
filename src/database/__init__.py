"""
Database Module
Handles SQLite database operations for storing PDF documents, summaries, and processing history
"""

from .models import (
    PDFDocument, Summary, ProcessingHistory, DatabaseManager,
    get_db_manager, initialize_database
)

__all__ = [
    "PDFDocument",
    "Summary",
    "ProcessingHistory",
    "DatabaseManager",
    "get_db_manager",
    "initialize_database"
]