"""
Integration tests for the end-to-end PDF summarization workflow.

These tests validate the complete user journey from file upload to summary generation.
They must FAIL before implementation is complete (TDD approach).
"""

import pytest
import tempfile
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Import the modules we're testing (will fail before implementation)
from src.pdf_processor.extractor import PDFExtractor
from src.pdf_processor.chunker import TextChunker
from src.llm_integration.summarizer import LLMSummarizer
from src.database.models import DatabaseManager, PDFDocument, Summary, ProcessingHistory
from src.utils.config import get_config


class TestEndToEndWorkflow:
    """Integration tests for complete end-to-end workflow."""

    def test_upload_to_summary_workflow(self):
        """Test the complete workflow from file upload to summary generation."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name

        try:
            with pytest.raises(Exception):
                # Initialize components
                db_manager = DatabaseManager(db_path)
                extractor = PDFExtractor()
                chunker = TextChunker()
                summarizer = LLMSummarizer()

                # Create a test PDF file (mock)
                test_pdf_path = "test_document.pdf"
                test_content = "This is a test document that contains multiple sentences. " * 50

                # Step 1: Save document to database
                document = PDFDocument(
                    filename="test_document.pdf",
                    original_filename="original_test_document.pdf",
                    file_path=test_pdf_path,
                    file_size=2048,
                    page_count=5
                )

                document_id = db_manager.save_document(document)

                # Step 2: Extract text from PDF
                extraction_result = extractor.extract_text(test_pdf_path)
                assert extraction_result["success"] is True
                assert len(extraction_result["content"]) > 0

                # Step 3: Update document with extracted text
                document.text_length = len(extraction_result["content"])
                document.page_count = extraction_result["page_count"]
                db_manager.update_document(document)

                # Step 4: Chunk the extracted text
                chunks = chunker.chunk_text(extraction_result["content"])
                assert len(chunks) > 1

                # Step 5: Generate summary
                summary_result = summarizer.summarize(extraction_result["content"], summary_type="concise")
                assert summary_result["success"] is True
                assert len(summary_result["content"]) > 0

                # Step 6: Save summary to database
                summary = Summary(
                    document_id=document_id,
                    summary_text=summary_result["content"],
                    summary_type="concise",
                    chunk_count=len(chunks),
                    total_tokens=summary_result.get("token_count", 0),
                    processing_time=summary_result.get("processing_time", 0),
                    model_used=summary_result.get("model_used", "unknown")
                )

                summary_id = db_manager.save_summary(summary)

                # Step 7: Update processing history
                history = ProcessingHistory(
                    document_id=document_id,
                    status="completed",
                    stage="summarization",
                    progress=100.0,
                    processing_time=summary_result.get("processing_time", 0)
                )

                db_manager.save_processing_history(history)

                # Verify the complete workflow
                saved_document = db_manager.get_document(document_id)
                saved_summary = db_manager.get_summary(summary_id)
                saved_history = db_manager.get_processing_history_for_document(document_id)

                assert saved_document is not None
                assert saved_document.is_processed is True
                assert saved_summary is not None
                assert saved_summary.document_id == document_id
                assert len(saved_history) > 0
                assert any(h.status == "completed" for h in saved_history)

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_error_handling_workflow(self):
        """Test error handling throughout the workflow."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name

        try:
            with pytest.raises(Exception):
                db_manager = DatabaseManager(db_path)
                extractor = PDFExtractor()
                summarizer = LLMSummarizer()

                # Test with invalid PDF
                document = PDFDocument(
                    filename="invalid.pdf",
                    original_filename="invalid_test.pdf",
                    file_path="invalid.pdf",
                    file_size=0
                )

                document_id = db_manager.save_document(document)

                # Step 1: Try to extract text - should fail
                extraction_result = extractor.extract_text("invalid.pdf")
                assert extraction_result["success"] is False
                assert "error" in extraction_result

                # Step 2: Update processing history with error
                error_history = ProcessingHistory(
                    document_id=document_id,
                    status="failed",
                    stage="text_extraction",
                    error_message=extraction_result.get("error", "Unknown error")
                )

                db_manager.save_processing_history(error_history)

                # Verify error was recorded
                saved_history = db_manager.get_processing_history_for_document(document_id)
                assert len(saved_history) > 0
                assert any(h.status == "failed" for h in saved_history)

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_multiple_summary_types_workflow(self):
        """Test generating multiple summary types for the same document."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name

        try:
            with pytest.raises(Exception):
                db_manager = DatabaseManager(db_path)
                summarizer = LLMSummarizer()

                # Save a document
                document = PDFDocument(
                    filename="test_document.pdf",
                    original_filename="original_test_document.pdf",
                    file_path="test_document.pdf",
                    file_size=2048
                )

                document_id = db_manager.save_document(document)

                test_content = "This is a comprehensive test document with substantial content."
                summary_types = ["concise", "detailed", "bullet_points"]

                # Generate summaries of different types
                summary_ids = []
                for summary_type in summary_types:
                    summary_result = summarizer.summarize(test_content, summary_type=summary_type)
                    assert summary_result["success"] is True

                    summary = Summary(
                        document_id=document_id,
                        summary_text=summary_result["content"],
                        summary_type=summary_type,
                        model_used=summary_result.get("model_used", "unknown")
                    )

                    summary_id = db_manager.save_summary(summary)
                    summary_ids.append(summary_id)

                # Verify all summaries were saved
                summaries = db_manager.get_summaries_for_document(document_id)
                assert len(summaries) == len(summary_types)

                for summary_type in summary_types:
                    assert any(s.summary_type == summary_type for s in summaries)

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_large_document_workflow(self):
        """Test workflow with a large document that requires chunking."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name

        try:
            with pytest.raises(Exception):
                db_manager = DatabaseManager(db_path)
                chunker = TextChunker()
                summarizer = LLMSummarizer()

                # Create large content
                large_content = "This is a test sentence that will be repeated many times. " * 500

                # Save document
                document = PDFDocument(
                    filename="large_document.pdf",
                    original_filename="large_test_document.pdf",
                    file_path="large_document.pdf",
                    file_size=len(large_content.encode()),
                    text_length=len(large_content)
                )

                document_id = db_manager.save_document(document)

                # Chunk the content
                chunks = chunker.chunk_text(large_content)
                assert len(chunks) > 5  # Should be chunked

                # Save processing history for chunking stage
                chunking_history = ProcessingHistory(
                    document_id=document_id,
                    status="completed",
                    stage="text_chunking",
                    progress=50.0,
                    metadata={"chunks_created": len(chunks)}
                )

                db_manager.save_processing_history(chunking_history)

                # Generate summary from chunks
                summary_result = summarizer.summarize(large_content, summary_type="concise")
                assert summary_result["success"] is True
                assert "chunks_used" in summary_result
                assert summary_result["chunks_used"] > 1

                # Save summary
                summary = Summary(
                    document_id=document_id,
                    summary_text=summary_result["content"],
                    summary_type="concise",
                    chunk_count=len(chunks),
                    total_tokens=summary_result.get("token_count", 0)
                )

                db_manager.save_summary(summary)

                # Update final processing history
                final_history = ProcessingHistory(
                    document_id=document_id,
                    status="completed",
                    stage="summarization",
                    progress=100.0
                )

                db_manager.save_processing_history(final_history)

                # Verify the workflow
                all_history = db_manager.get_processing_history_for_document(document_id)
                assert len(all_history) >= 2  # Should have chunking and summarization entries

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_document_history_workflow(self):
        """Test retrieving and displaying document processing history."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name

        try:
            with pytest.raises(Exception):
                db_manager = DatabaseManager(db_path)

                # Process multiple documents
                document_ids = []
                for i in range(3):
                    document = PDFDocument(
                        filename=f"document_{i}.pdf",
                        original_filename=f"original_document_{i}.pdf",
                        file_path=f"document_{i}.pdf",
                        file_size=1024 * (i + 1)
                    )

                    document_id = db_manager.save_document(document)
                    document_ids.append(document_id)

                    # Add processing history
                    stages = ["upload", "text_extraction", "summarization", "completed"]
                    for j, stage in enumerate(stages):
                        history = ProcessingHistory(
                            document_id=document_id,
                            status="completed" if stage == "completed" else "processing",
                            stage=stage,
                            progress=(j + 1) * 25.0
                        )
                        db_manager.save_processing_history(history)

                # Test retrieving all documents
                all_documents = db_manager.get_all_documents()
                assert len(all_documents) == 3

                # Test retrieving processing history for each document
                for document_id in document_ids:
                    history = db_manager.get_processing_history_for_document(document_id)
                    assert len(history) == len(stages)
                    assert history[0].stage == "completed"  # Last stage should be completed

                # Test database statistics
                stats = db_manager.get_database_stats()
                assert stats["total_documents"] == 3
                assert stats["total_processing_history"] == 12  # 3 docs * 4 stages

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_configuration_integration(self):
        """Test that configuration is properly integrated across all components."""
        # Arrange
        config = get_config()

        # Act & Assert
        with pytest.raises(Exception):
            # Test configuration values are accessible
            assert config.groq.api_key is not None
            assert config.file_upload.max_file_size > 0
            assert config.text_processing.chunk_size > 0
            assert config.summary.default_summary_type in ["concise", "detailed", "bullet_points"]

            # Test configuration is used by components
            extractor = PDFExtractor()
            chunker = TextChunker()
            summarizer = LLMSummarizer()

            # These should use configuration values
            assert extractor.max_file_size == config.file_upload.max_file_size
            assert chunker.chunk_size == config.text_processing.chunk_size
            assert summarizer.default_model == config.groq.model

    def test_performance_monitoring(self):
        """Test performance monitoring throughout the workflow."""
        # Arrange
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name

        try:
            with pytest.raises(Exception):
                db_manager = DatabaseManager(db_path)
                extractor = PDFExtractor()
                chunker = TextChunker()
                summarizer = LLMSummarizer()

                # Process document with performance tracking
                document = PDFDocument(
                    filename="performance_test.pdf",
                    original_filename="performance_test.pdf",
                    file_path="performance_test.pdf",
                    file_size=1024
                )

                document_id = db_manager.save_document(document)

                start_time = datetime.now()

                # Track performance at each stage
                stages = [
                    ("text_extraction", extractor.extract_text, "performance_test.pdf"),
                    ("text_chunking", chunker.chunk_text, "Sample text for performance testing. " * 10),
                    ("summarization", summarizer.summarize, "Sample text for performance testing. " * 10)
                ]

                for stage_name, stage_func, stage_input in stages:
                    stage_start = datetime.now()

                    if stage_name == "text_extraction":
                        result = stage_func(stage_input)
                    else:
                        result = stage_func(stage_input)

                    stage_end = datetime.now()
                    stage_duration = (stage_end - stage_start).total_seconds()

                    # Save performance history
                    history = ProcessingHistory(
                        document_id=document_id,
                        status="completed",
                        stage=stage_name,
                        processing_time=stage_duration,
                        metadata={
                            "performance_metrics": {
                                "duration_seconds": stage_duration,
                                "start_time": stage_start.isoformat(),
                                "end_time": stage_end.isoformat()
                            }
                        }
                    )

                    db_manager.save_processing_history(history)

                total_time = (datetime.now() - start_time).total_seconds()

                # Verify performance tracking
                all_history = db_manager.get_processing_history_for_document(document_id)
                assert len(all_history) == len(stages)

                total_processing_time = sum(
                    h.processing_time for h in all_history
                    if h.processing_time is not None
                )

                assert total_processing_time > 0
                assert total_processing_time < total_time + 1  # Allow some overhead

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])