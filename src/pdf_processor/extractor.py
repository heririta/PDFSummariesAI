"""
PDF Text Extraction Module
Handles extraction of text from PDF files using PyMuPDF
"""

import fitz  # PyMuPDF
import os
import hashlib
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extracts text content from PDF files"""

    def __init__(self, config=None):
        self.logger = logger
        self.config = config

    def extract_text(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from a PDF file

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary with extraction results
        """
        try:
            # Validate file exists
            if not os.path.exists(pdf_path):
                return {
                    "success": False,
                    "error": "File not found",
                    "error_type": "file_not_found"
                }

            # Check file size
            file_size = os.path.getsize(pdf_path)
            if self.config and hasattr(self.config, 'file_upload'):
                max_size = self.config.file_upload.max_file_size
                if file_size > max_size:
                    return {
                        "success": False,
                        "error": f"File size ({file_size / (1024*1024):.1f} MB) exceeds maximum ({max_size / (1024*1024):.1f} MB)",
                        "error_type": "size_exceeded"
                    }

            # Open PDF document
            doc = fitz.open(pdf_path)

            # Check if PDF is password protected
            if doc.needs_pass:
                doc.close()
                return {
                    "success": False,
                    "error": "Password-protected PDFs are not supported",
                    "error_type": "password_protected"
                }

            # Extract text from all pages
            text_content = []
            total_pages = len(doc)

            for page_num in range(total_pages):
                try:
                    page = doc.load_page(page_num)
                    page_text = page.get_text()

                    if page_text.strip():
                        text_content.append(page_text)

                except Exception as e:
                    self.logger.warning(f"Error extracting text from page {page_num}: {e}")
                    continue

            doc.close()

            # Combine all text
            full_text = "\n\n".join(text_content).strip()

            if not full_text:
                return {
                    "success": False,
                    "error": "No extractable text found in PDF. The PDF might contain only images.",
                    "error_type": "no_text_content"
                }

            return {
                "success": True,
                "content": full_text,
                "page_count": total_pages,
                "file_size": file_size,
                "text_length": len(full_text),
                "extraction_method": "pymupdf"
            }

        except fitz.FileDataError:
            return {
                "success": False,
                "error": "Corrupted or invalid PDF file",
                "error_type": "corrupted"
            }

        except Exception as e:
            self.logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "error_type": "extraction_error"
            }

    def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a PDF file

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary containing PDF metadata
        """
        try:
            if not os.path.exists(pdf_path):
                return {"success": False, "error": "File not found"}

            doc = fitz.open(pdf_path)

            if doc.needs_pass:
                doc.close()
                return {"success": False, "error": "Password-protected PDF"}

            # Get basic metadata
            metadata = doc.metadata
            page_count = len(doc)

            # Calculate file hash
            file_hash = self._calculate_file_hash(pdf_path)

            # Get file size
            file_size = os.path.getsize(pdf_path)

            doc.close()

            return {
                "success": True,
                "metadata": {
                    "title": metadata.get("title", ""),
                    "author": metadata.get("author", ""),
                    "subject": metadata.get("subject", ""),
                    "creator": metadata.get("creator", ""),
                    "producer": metadata.get("producer", ""),
                    "creation_date": metadata.get("creationDate", ""),
                    "modification_date": metadata.get("modDate", ""),
                    "page_count": page_count,
                    "file_size": file_size,
                    "file_hash": file_hash
                }
            }

        except Exception as e:
            self.logger.error(f"Error extracting metadata from PDF {pdf_path}: {e}")
            return {
                "success": False,
                "error": f"Metadata extraction failed: {str(e)}"
            }

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of the file"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating file hash: {e}")
            return ""

    def validate_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Validate PDF file and return validation results

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary with validation results
        """
        try:
            if not os.path.exists(pdf_path):
                return {
                    "valid": False,
                    "error": "File does not exist",
                    "error_type": "file_not_found"
                }

            # Check file extension
            if not pdf_path.lower().endswith('.pdf'):
                return {
                    "valid": False,
                    "error": "File must have .pdf extension",
                    "error_type": "invalid_extension"
                }

            # Try to open with PyMuPDF
            doc = fitz.open(pdf_path)

            # Check if password protected
            if doc.needs_pass:
                doc.close()
                return {
                    "valid": False,
                    "error": "Password-protected PDFs are not supported",
                    "error_type": "password_protected"
                }

            # Check if PDF has pages
            page_count = len(doc)
            if page_count == 0:
                doc.close()
                return {
                    "valid": False,
                    "error": "PDF has no pages",
                    "error_type": "no_pages"
                }

            doc.close()

            return {
                "valid": True,
                "page_count": page_count,
                "file_size": os.path.getsize(pdf_path)
            }

        except fitz.FileDataError:
            return {
                "valid": False,
                "error": "Corrupted or invalid PDF file",
                "error_type": "corrupted"
            }

        except Exception as e:
            return {
                "valid": False,
                "error": f"Validation error: {str(e)}",
                "error_type": "validation_error"
            }