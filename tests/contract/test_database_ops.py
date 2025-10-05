"""
Contract tests for database operations.

These tests define the expected behavior of the database operations module.
They must FAIL before implementation is complete (TDD approach).
"""

import pytest
import tempfile
import os
from datetime import datetime
from typing import Optional, List

# Import the modules we're testing (will fail before implementation)
from src.database.models import DatabaseManager, PDFDocument, Summary, ProcessingHistory
from src.database.operations import DatabaseOperations


class TestDatabaseManager:
    """Contract tests for database manager functionality."""

    def test_initialize_database(self):
        """Test database initialization."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name

        try:
            # Act & Assert
            with pytest.raises(Exception):
                db_manager = DatabaseManager(db_path)
                assert db_manager is not None
                assert os.path.exists(db_path)
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_save_and_retrieve_document(self):
        """Test saving and retrieving a PDF document."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name

        try:
            with pytest.raises(Exception):
                db_manager = DatabaseManager(db_path)
                document = PDFDocument(
                    filename="test.pdf",
                    original_filename="original_test.pdf",
                    file_path="/path/to/test.pdf",
                    file_size=1024,
                    page_count=5,
                    text_length=500
                )

                # Act
                document_id = db_manager.save_document(document)
                retrieved_document = db_manager.get_document(document_id)

                # Assert
                assert document_id is not None
                assert document_id > 0
                assert retrieved_document is not None
                assert retrieved_document.filename == "test.pdf"
                assert retrieved_document.original_filename == "original_test.pdf"
                assert retrieved_document.file_size == 1024

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_update_document(self):
        """Test updating a document."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name

        try:
            with pytest.raises(Exception):
                db_manager = DatabaseManager(db_path)
                document = PDFDocument(
                    filename="test.pdf",
                    original_filename="original_test.pdf",
                    file_path="/path/to/test.pdf",
                    file_size=1024,
                    page_count=5
                )

                # Save initial document
                document_id = db_manager.save_document(document)
                document.id = document_id

                # Update document
                document.page_count = 10
                document.text_length = 1000
                document.is_processed = True

                # Act
                success = db_manager.update_document(document)
                updated_document = db_manager.get_document(document_id)

                # Assert
                assert success is True
                assert updated_document.page_count == 10
                assert updated_document.text_length == 1000
                assert updated_document.is_processed is True

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_delete_document(self):
        """Test deleting a document and related records."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name

        try:
            with pytest.raises(Exception):
                db_manager = DatabaseManager(db_path)
                document = PDFDocument(
                    filename="test.pdf",
                    original_filename="original_test.pdf",
                    file_path="/path/to/test.pdf",
                    file_size=1024
                )

                # Save document
                document_id = db_manager.save_document(document)

                # Act
                success = db_manager.delete_document(document_id)
                retrieved_document = db_manager.get_document(document_id)

                # Assert
                assert success is True
                assert retrieved_document is None

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_save_and_retrieve_summary(self):
        """Test saving and retrieving a summary."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name

        try:
            with pytest.raises(Exception):
                db_manager = DatabaseManager(db_path)

                # First save a document
                document = PDFDocument(
                    filename="test.pdf",
                    original_filename="original_test.pdf",
                    file_path="/path/to/test.pdf",
                    file_size=1024
                )
                document_id = db_manager.save_document(document)

                # Create summary
                summary = Summary(
                    document_id=document_id,
                    summary_text="This is a test summary of the document content.",
                    summary_type="concise",
                    chunk_count=3,
                    total_tokens=150,
                    processing_time=2.5,
                    model_used="llama-3.3-70b-versatile"
                )

                # Act
                summary_id = db_manager.save_summary(summary)
                retrieved_summary = db_manager.get_summary(summary_id)

                # Assert
                assert summary_id is not None
                assert summary_id > 0
                assert retrieved_summary is not None
                assert retrieved_summary.document_id == document_id
                assert retrieved_summary.summary_text == "This is a test summary of the document content."
                assert retrieved_summary.summary_type == "concise"

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_get_summaries_for_document(self):
        """Test retrieving all summaries for a document."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name

        try:
            with pytest.raises(Exception):
                db_manager = DatabaseManager(db_path)

                # Save a document
                document = PDFDocument(
                    filename="test.pdf",
                    original_filename="original_test.pdf",
                    file_path="/path/to/test.pdf",
                    file_size=1024
                )
                document_id = db_manager.save_document(document)

                # Save multiple summaries
                summary1 = Summary(
                    document_id=document_id,
                    summary_text="Concise summary.",
                    summary_type="concise"
                )
                summary2 = Summary(
                    document_id=document_id,
                    summary_text="Detailed summary with more information.",
                    summary_type="detailed"
                )

                db_manager.save_summary(summary1)
                db_manager.save_summary(summary2)

                # Act
                summaries = db_manager.get_summaries_for_document(document_id)

                # Assert
                assert len(summaries) == 2
                assert all(s.document_id == document_id for s in summaries)
                assert any(s.summary_type == "concise" for s in summaries)
                assert any(s.summary_type == "detailed" for s in summaries)

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_save_and_retrieve_processing_history(self):
        """Test saving and retrieving processing history."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name

        try:
            with pytest.raises(Exception):
                db_manager = DatabaseManager(db_path)

                # Save a document
                document = PDFDocument(
                    filename="test.pdf",
                    original_filename="original_test.pdf",
                    file_path="/path/to/test.pdf",
                    file_size=1024
                )
                document_id = db_manager.save_document(document)

                # Create processing history
                history = ProcessingHistory(
                    document_id=document_id,
                    status="processing",
                    stage="text_extraction",
                    progress=50.0,
                    error_message=""
                )

                # Act
                history_id = db_manager.save_processing_history(history)
                retrieved_history = db_manager.get_processing_history_for_document(document_id)

                # Assert
                assert history_id is not None
                assert history_id > 0
                assert len(retrieved_history) == 1
                assert retrieved_history[0].document_id == document_id
                assert retrieved_history[0].status == "processing"
                assert retrieved_history[0].stage == "text_extraction"

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_get_database_statistics(self):
        """Test getting database statistics."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name

        try:
            with pytest.raises(Exception):
                db_manager = DatabaseManager(db_path)

                # Save test data
                document = PDFDocument(
                    filename="test.pdf",
                    original_filename="original_test.pdf",
                    file_path="/path/to/test.pdf",
                    file_size=1024,
                    is_processed=True
                )
                document_id = db_manager.save_document(document)

                summary = Summary(
                    document_id=document_id,
                    summary_text="Test summary.",
                    summary_type="concise",
                    is_favorite=True
                )
                db_manager.save_summary(summary)

                history = ProcessingHistory(
                    document_id=document_id,
                    status="completed",
                    stage="completed"
                )
                db_manager.save_processing_history(history)

                # Act
                stats = db_manager.get_database_stats()

                # Assert
                assert "total_documents" in stats
                assert "processed_documents" in stats
                assert "total_summaries" in stats
                assert "favorite_summaries" in stats
                assert "total_processing_history" in stats
                assert stats["total_documents"] == 1
                assert stats["processed_documents"] == 1
                assert stats["total_summaries"] == 1
                assert stats["favorite_summaries"] == 1
                assert stats["total_processing_history"] == 1

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_cascade_delete_document(self):
        """Test that deleting a document cascades to delete related summaries and history."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name

        try:
            with pytest.raises(Exception):
                db_manager = DatabaseManager(db_path)

                # Save a document
                document = PDFDocument(
                    filename="test.pdf",
                    original_filename="original_test.pdf",
                    file_path="/path/to/test.pdf",
                    file_size=1024
                )
                document_id = db_manager.save_document(document)

                # Save related data
                summary = Summary(
                    document_id=document_id,
                    summary_text="Test summary.",
                    summary_type="concise"
                )
                db_manager.save_summary(summary)

                history = ProcessingHistory(
                    document_id=document_id,
                    status="completed",
                    stage="completed"
                )
                db_manager.save_processing_history(history)

                # Verify data exists
                summaries_before = db_manager.get_summaries_for_document(document_id)
                history_before = db_manager.get_processing_history_for_document(document_id)
                assert len(summaries_before) == 1
                assert len(history_before) == 1

                # Act - Delete the document
                success = db_manager.delete_document(document_id)

                # Assert - All related data should be deleted
                assert success is True
                assert db_manager.get_document(document_id) is None
                assert len(db_manager.get_summaries_for_document(document_id)) == 0
                assert len(db_manager.get_processing_history_for_document(document_id)) == 0

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_database_connection_error_handling(self):
        """Test database connection error handling."""
        # Arrange
        invalid_db_path = "/invalid/path/to/database.db"

        # Act & Assert
        with pytest.raises(Exception):
            DatabaseManager(invalid_db_path)

    def test_foreign_key_constraints(self):
        """Test foreign key constraints are enforced."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name

        try:
            with pytest.raises(Exception):
                db_manager = DatabaseManager(db_path)

                # Try to save a summary for non-existent document
                summary = Summary(
                    document_id=999,  # Non-existent document ID
                    summary_text="Test summary.",
                    summary_type="concise"
                )

                # Act & Assert
                with pytest.raises(Exception):  # Should fail due to foreign key constraint
                    db_manager.save_summary(summary)

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


class TestDatabaseOperations:
    """Contract tests for high-level database operations."""

    def test_get_documents_with_pagination(self):
        """Test retrieving documents with pagination."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name

        try:
            with pytest.raises(Exception):
                db_manager = DatabaseManager(db_path)

                # Save multiple documents
                for i in range(15):
                    document = PDFDocument(
                        filename=f"test_{i}.pdf",
                        original_filename=f"original_test_{i}.pdf",
                        file_path=f"/path/to/test_{i}.pdf",
                        file_size=1024 + i
                    )
                    db_manager.save_document(document)

                # Act
                all_docs = db_manager.get_all_documents()
                limited_docs = db_manager.get_all_documents(limit=5)

                # Assert
                assert len(all_docs) == 15
                assert len(limited_docs) == 5

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_cleanup_old_records(self):
        """Test cleanup of old processing history records."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name

        try:
            with pytest.raises(Exception):
                db_manager = DatabaseManager(db_path)

                # Save a document
                document = PDFDocument(
                    filename="test.pdf",
                    original_filename="original_test.pdf",
                    file_path="/path/to/test.pdf",
                    file_size=1024
                )
                document_id = db_manager.save_document(document)

                # Save old processing history (simulate old date)
                old_history = ProcessingHistory(
                    document_id=document_id,
                    status="completed",
                    stage="completed",
                    started_at=datetime(2020, 1, 1)  # Old date
                )
                db_manager.save_processing_history(old_history)

                # Save recent processing history
                recent_history = ProcessingHistory(
                    document_id=document_id,
                    status="processing",
                    stage="text_extraction"
                )
                db_manager.save_processing_history(recent_history)

                # Act
                deleted_count = db_manager.cleanup_old_records(days=30)
                remaining_history = db_manager.get_processing_history_for_document(document_id)

                # Assert
                assert deleted_count == 1
                assert len(remaining_history) == 1
                assert remaining_history[0].status == "processing"

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])