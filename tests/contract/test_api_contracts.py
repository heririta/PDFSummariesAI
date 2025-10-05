"""
Contract Tests for API Interfaces
Tests that all modules adhere to their defined contracts
"""

import pytest
from pdf_processor.extractor import PDFExtractor
from pdf_processor.chunker import TextChunker
from llm_integration.summarizer import GroqSummarizer
from llm_integration.pipeline import SummarizationPipeline
from database.models import DatabaseManager, PDFSummary


class TestPDFExtractorContract:
    """Contract tests for PDFExtractor"""

    def test_extract_text_method_signature(self):
        """Test extract_text method has correct signature"""
        extractor = PDFExtractor()
        assert hasattr(extractor, 'extract_text')
        assert callable(extractor.extract_text)

    def test_extract_metadata_method_signature(self):
        """Test extract_metadata method has correct signature"""
        extractor = PDFExtractor()
        assert hasattr(extractor, 'extract_metadata')
        assert callable(extractor.extract_metadata)


class TestTextChunkerContract:
    """Contract tests for TextChunker"""

    def test_chunk_text_method_signature(self):
        """Test chunk_text method has correct signature"""
        chunker = TextChunker()
        assert hasattr(chunker, 'chunk_text')
        assert callable(chunker.chunk_text)

    def test_clean_text_method_signature(self):
        """Test clean_text method has correct signature"""
        chunker = TextChunker()
        assert hasattr(chunker, 'clean_text')
        assert callable(chunker.clean_text)


class TestGroqSummarizerContract:
    """Contract tests for GroqSummarizer"""

    def test_summarize_text_method_signature(self, mock_groq_api_key):
        """Test summarize_text method has correct signature"""
        summarizer = GroqSummarizer(mock_groq_api_key)
        assert hasattr(summarizer, 'summarize_text')
        assert callable(summarizer.summarize_text)

    def test_summarize_chunks_method_signature(self, mock_groq_api_key):
        """Test summarize_chunks method has correct signature"""
        summarizer = GroqSummarizer(mock_groq_api_key)
        assert hasattr(summarizer, 'summarize_chunks')
        assert callable(summarizer.summarize_chunks)


class TestSummarizationPipelineContract:
    """Contract tests for SummarizationPipeline"""

    def test_process_pdf_method_signature(self, mock_groq_api_key):
        """Test process_pdf method has correct signature"""
        pipeline = SummarizationPipeline(mock_groq_api_key)
        assert hasattr(pipeline, 'process_pdf')
        assert callable(pipeline.process_pdf)

    def test_process_text_method_signature(self, mock_groq_api_key):
        """Test process_text method has correct signature"""
        pipeline = SummarizationPipeline(mock_groq_api_key)
        assert hasattr(pipeline, 'process_text')
        assert callable(pipeline.process_text)


class TestDatabaseManagerContract:
    """Contract tests for DatabaseManager"""

    def test_save_summary_method_signature(self):
        """Test save_summary method has correct signature"""
        db_manager = DatabaseManager(":memory:")
        assert hasattr(db_manager, 'save_summary')
        assert callable(db_manager.save_summary)

    def test_get_summary_method_signature(self):
        """Test get_summary method has correct signature"""
        db_manager = DatabaseManager(":memory:")
        assert hasattr(db_manager, 'get_summary')
        assert callable(db_manager.get_summary)

    def test_get_all_summaries_method_signature(self):
        """Test get_all_summaries method has correct signature"""
        db_manager = DatabaseManager(":memory:")
        assert hasattr(db_manager, 'get_all_summaries')
        assert callable(db_manager.get_all_summaries)

    def test_delete_summary_method_signature(self):
        """Test delete_summary method has correct signature"""
        db_manager = DatabaseManager(":memory:")
        assert hasattr(db_manager, 'delete_summary')
        assert callable(db_manager.delete_summary)


class TestPDFSummaryContract:
    """Contract tests for PDFSummary"""

    def test_to_dict_method_signature(self):
        """Test to_dict method has correct signature"""
        summary = PDFSummary()
        assert hasattr(summary, 'to_dict')
        assert callable(summary.to_dict)