"""
Database Models Module

Defines database models and operations for PDF Summarizer AI application.
Implements three main entities: PDF Document, Summary, and Processing History.
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
import logging
from enum import Enum

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import get_config

logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Processing status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SummaryType(Enum):
    """Summary type enumeration"""
    CONCISE = "concise"
    DETAILED = "detailed"
    BULLET_POINTS = "bullet_points"
    EXECUTIVE = "executive"


class PDFDocument:
    """Represents a PDF document record"""

    def __init__(
        self,
        id: Optional[int] = None,
        filename: str = "",
        original_filename: str = "",
        file_path: str = "",
        file_size: int = 0,
        page_count: int = 0,
        text_length: int = 0,
        upload_date: Optional[datetime] = None,
        file_hash: str = "",
        mime_type: str = "application/pdf",
        is_processed: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.id = id
        self.filename = filename
        self.original_filename = original_filename
        self.file_path = file_path
        self.file_size = file_size
        self.page_count = page_count
        self.text_length = text_length
        self.upload_date = upload_date or datetime.now()
        self.file_hash = file_hash
        self.mime_type = mime_type
        self.is_processed = is_processed
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "filename": self.filename,
            "original_filename": self.original_filename,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "page_count": self.page_count,
            "text_length": self.text_length,
            "upload_date": self.upload_date.isoformat() if self.upload_date else None,
            "file_hash": self.file_hash,
            "mime_type": self.mime_type,
            "is_processed": self.is_processed,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PDFDocument":
        """Create from dictionary"""
        if data.get("upload_date"):
            data["upload_date"] = datetime.fromisoformat(data["upload_date"])
        return cls(**data)


class Summary:
    """Represents a summary record"""

    def __init__(
        self,
        id: Optional[int] = None,
        document_id: int = 0,
        summary_text: str = "",
        summary_type: str = SummaryType.CONCISE.value,
        chunk_count: int = 0,
        total_tokens: int = 0,
        processing_time: float = 0.0,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        model_used: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1000,
        is_favorite: bool = False,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.id = id
        self.document_id = document_id
        self.summary_text = summary_text
        self.summary_type = summary_type
        self.chunk_count = chunk_count
        self.total_tokens = total_tokens
        self.processing_time = processing_time
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()
        self.model_used = model_used
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.is_favorite = is_favorite
        self.tags = tags or []
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "summary_text": self.summary_text,
            "summary_type": self.summary_type,
            "chunk_count": self.chunk_count,
            "total_tokens": self.total_tokens,
            "processing_time": self.processing_time,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "model_used": self.model_used,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "is_favorite": self.is_favorite,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Summary":
        """Create from dictionary"""
        if data.get("created_at"):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("updated_at"):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


class ProcessingHistory:
    """Represents a processing history record"""

    def __init__(
        self,
        id: Optional[int] = None,
        document_id: int = 0,
        status: str = ProcessingStatus.PENDING.value,
        stage: str = "",
        progress: float = 0.0,
        error_message: str = "",
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        processing_time: float = 0.0,
        retry_count: int = 0,
        max_retries: int = 3,
        log_entries: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.id = id
        self.document_id = document_id
        self.status = status
        self.stage = stage
        self.progress = progress
        self.error_message = error_message
        self.started_at = started_at or datetime.now()
        self.completed_at = completed_at
        self.processing_time = processing_time
        self.retry_count = retry_count
        self.max_retries = max_retries
        self.log_entries = log_entries or []
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "processing_time": self.processing_time,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "log_entries": self.log_entries,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingHistory":
        """Create from dictionary"""
        if data.get("started_at"):
            data["started_at"] = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            data["completed_at"] = datetime.fromisoformat(data["completed_at"])
        return cls(**data)


class DatabaseManager:
    """
    Manages SQLite database operations for PDF Summarizer AI.

    Handles CRUD operations for PDF Document, Summary, and Processing History entities.
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize database manager"""
        config = get_config()
        self.db_path = db_path or config.database.path
        self.logger = logging.getLogger(__name__)
        self._init_database()
        self._create_indexes()

    def _init_database(self):
        """Initialize database tables"""
        try:
            # Ensure database directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")

                # Create pdf_documents table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS pdf_documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT NOT NULL,
                        original_filename TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        file_size INTEGER DEFAULT 0,
                        page_count INTEGER DEFAULT 0,
                        text_length INTEGER DEFAULT 0,
                        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        file_hash TEXT UNIQUE,
                        mime_type TEXT DEFAULT 'application/pdf',
                        is_processed BOOLEAN DEFAULT 0,
                        metadata TEXT DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Create summaries table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS summaries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_id INTEGER NOT NULL,
                        summary_text TEXT NOT NULL,
                        summary_type TEXT DEFAULT 'concise',
                        chunk_count INTEGER DEFAULT 0,
                        total_tokens INTEGER DEFAULT 0,
                        processing_time REAL DEFAULT 0.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        model_used TEXT DEFAULT '',
                        temperature REAL DEFAULT 0.3,
                        max_tokens INTEGER DEFAULT 1000,
                        is_favorite BOOLEAN DEFAULT 0,
                        tags TEXT DEFAULT '[]',
                        metadata TEXT DEFAULT '{}',
                        FOREIGN KEY (document_id) REFERENCES pdf_documents (id) ON DELETE CASCADE
                    )
                """)

                # Create processing_history table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS processing_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_id INTEGER NOT NULL,
                        status TEXT DEFAULT 'pending',
                        stage TEXT DEFAULT '',
                        progress REAL DEFAULT 0.0,
                        error_message TEXT DEFAULT '',
                        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        completed_at TIMESTAMP,
                        processing_time REAL DEFAULT 0.0,
                        retry_count INTEGER DEFAULT 0,
                        max_retries INTEGER DEFAULT 3,
                        log_entries TEXT DEFAULT '[]',
                        metadata TEXT DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (document_id) REFERENCES pdf_documents (id) ON DELETE CASCADE
                    )
                """)

                conn.commit()
                self.logger.info("Database tables initialized successfully")

        except sqlite3.Error as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise

    def _create_indexes(self):
        """Create database indexes for better performance"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Indexes for pdf_documents
                conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_filename ON pdf_documents(filename)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_upload_date ON pdf_documents(upload_date)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON pdf_documents(file_hash)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_is_processed ON pdf_documents(is_processed)")

                # Indexes for summaries
                conn.execute("CREATE INDEX IF NOT EXISTS idx_summaries_document_id ON summaries(document_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_summaries_summary_type ON summaries(summary_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_summaries_created_at ON summaries(created_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_summaries_is_favorite ON summaries(is_favorite)")

                # Indexes for processing_history
                conn.execute("CREATE INDEX IF NOT EXISTS idx_history_document_id ON processing_history(document_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_history_status ON processing_history(status)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_history_started_at ON processing_history(started_at)")

                conn.commit()
                self.logger.info("Database indexes created successfully")

        except sqlite3.Error as e:
            self.logger.error(f"Failed to create database indexes: {e}")
            raise

    def get_connection(self) -> sqlite3.Connection:
        """Get database connection with proper settings"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row  # Enable dictionary-like access
        return conn

    # PDF Document operations
    def save_document(self, document: PDFDocument) -> int:
        """Save a PDF document to the database"""
        try:
            with self.get_connection() as conn:
                # Check if document with same file_hash already exists
                cursor = conn.execute("SELECT id FROM pdf_documents WHERE file_hash = ?", (document.file_hash,))
                existing_doc = cursor.fetchone()

                if existing_doc:
                    # Update existing document
                    doc_id = existing_doc[0]
                    cursor = conn.execute("""
                        UPDATE pdf_documents SET
                            filename = ?, original_filename = ?, file_path = ?, file_size = ?,
                            page_count = ?, text_length = ?, mime_type = ?, is_processed = ?,
                            metadata = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (
                        document.filename,
                        document.original_filename,
                        document.file_path,
                        document.file_size,
                        document.page_count,
                        document.text_length,
                        document.mime_type,
                        document.is_processed,
                        json.dumps(document.metadata),
                        doc_id
                    ))
                    conn.commit()
                    self.logger.info(f"Updated existing document with ID: {doc_id}")
                    return doc_id
                else:
                    # Insert new document
                    cursor = conn.execute("""
                        INSERT INTO pdf_documents (
                            filename, original_filename, file_path, file_size, page_count,
                            text_length, file_hash, mime_type, is_processed, metadata, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (
                        document.filename,
                        document.original_filename,
                        document.file_path,
                        document.file_size,
                        document.page_count,
                        document.text_length,
                        document.file_hash,
                        document.mime_type,
                        document.is_processed,
                        json.dumps(document.metadata)
                    ))

                    document_id = cursor.lastrowid
                    conn.commit()
                    self.logger.info(f"Saved new document with ID: {document_id}")
                    return document_id

        except sqlite3.Error as e:
            self.logger.error(f"Failed to save document: {e}")
            raise

    def get_document(self, document_id: int) -> Optional[PDFDocument]:
        """Retrieve a PDF document by ID"""
        try:
            with self.get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM pdf_documents WHERE id = ?", (document_id,)
                ).fetchone()

                if row:
                    return PDFDocument(
                        id=row["id"],
                        filename=row["filename"],
                        original_filename=row["original_filename"],
                        file_path=row["file_path"],
                        file_size=row["file_size"],
                        page_count=row["page_count"],
                        text_length=row["text_length"],
                        upload_date=datetime.fromisoformat(row["upload_date"]) if row["upload_date"] else None,
                        file_hash=row["file_hash"],
                        mime_type=row["mime_type"],
                        is_processed=bool(row["is_processed"]),
                        metadata=json.loads(row["metadata"])
                    )
                return None

        except sqlite3.Error as e:
            self.logger.error(f"Failed to get document {document_id}: {e}")
            raise

    def get_all_documents(self, limit: Optional[int] = None) -> List[PDFDocument]:
        """Retrieve all PDF documents"""
        try:
            with self.get_connection() as conn:
                query = "SELECT * FROM pdf_documents ORDER BY upload_date DESC"
                if limit:
                    query += f" LIMIT {limit}"

                rows = conn.execute(query).fetchall()

                return [PDFDocument(
                    id=row["id"],
                    filename=row["filename"],
                    original_filename=row["original_filename"],
                    file_path=row["file_path"],
                    file_size=row["file_size"],
                    page_count=row["page_count"],
                    text_length=row["text_length"],
                    upload_date=datetime.fromisoformat(row["upload_date"]) if row["upload_date"] else None,
                    file_hash=row["file_hash"],
                    mime_type=row["mime_type"],
                    is_processed=bool(row["is_processed"]),
                    metadata=json.loads(row["metadata"])
                ) for row in rows]

        except sqlite3.Error as e:
            self.logger.error(f"Failed to get documents: {e}")
            raise

    def update_document(self, document: PDFDocument) -> bool:
        """Update a PDF document"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("""
                    UPDATE pdf_documents SET
                        filename = ?, original_filename = ?, file_path = ?, file_size = ?,
                        page_count = ?, text_length = ?, is_processed = ?, metadata = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (
                    document.filename,
                    document.original_filename,
                    document.file_path,
                    document.file_size,
                    document.page_count,
                    document.text_length,
                    document.is_processed,
                    json.dumps(document.metadata),
                    document.id
                ))

                success = cursor.rowcount > 0
                conn.commit()

                if success:
                    self.logger.info(f"Updated document {document.id}")
                return success

        except sqlite3.Error as e:
            self.logger.error(f"Failed to update document {document.id}: {e}")
            raise

    def delete_document(self, document_id: int) -> bool:
        """Delete a PDF document (and related summaries/history)"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM pdf_documents WHERE id = ?", (document_id,)
                )

                success = cursor.rowcount > 0
                conn.commit()

                if success:
                    self.logger.info(f"Deleted document {document_id}")
                return success

        except sqlite3.Error as e:
            self.logger.error(f"Failed to delete document {document_id}: {e}")
            raise

    # Summary operations
    def save_summary(self, summary: Summary) -> int:
        """Save a summary to the database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO summaries (
                        document_id, summary_text, summary_type, chunk_count, total_tokens,
                        processing_time, model_used, temperature, max_tokens, is_favorite,
                        tags, metadata, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    summary.document_id,
                    summary.summary_text,
                    summary.summary_type,
                    summary.chunk_count,
                    summary.total_tokens,
                    summary.processing_time,
                    summary.model_used,
                    summary.temperature,
                    summary.max_tokens,
                    summary.is_favorite,
                    json.dumps(summary.tags),
                    json.dumps(summary.metadata)
                ))

                summary_id = cursor.lastrowid
                conn.commit()
                self.logger.info(f"Saved summary for document {summary.document_id} with ID {summary_id}")
                return summary_id

        except sqlite3.Error as e:
            self.logger.error(f"Failed to save summary: {e}")
            raise

    def get_summary(self, summary_id: int) -> Optional[Summary]:
        """Retrieve a summary by ID"""
        try:
            with self.get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM summaries WHERE id = ?", (summary_id,)
                ).fetchone()

                if row:
                    return Summary(
                        id=row["id"],
                        document_id=row["document_id"],
                        summary_text=row["summary_text"],
                        summary_type=row["summary_type"],
                        chunk_count=row["chunk_count"],
                        total_tokens=row["total_tokens"],
                        processing_time=row["processing_time"],
                        created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
                        updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
                        model_used=row["model_used"],
                        temperature=row["temperature"],
                        max_tokens=row["max_tokens"],
                        is_favorite=bool(row["is_favorite"]),
                        tags=json.loads(row["tags"]),
                        metadata=json.loads(row["metadata"])
                    )
                return None

        except sqlite3.Error as e:
            self.logger.error(f"Failed to get summary {summary_id}: {e}")
            raise

    def get_summaries_for_document(self, document_id: int) -> List[Summary]:
        """Retrieve all summaries for a document"""
        try:
            with self.get_connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM summaries WHERE document_id = ? ORDER BY created_at DESC",
                    (document_id,)
                ).fetchall()

                return [Summary(
                    id=row["id"],
                    document_id=row["document_id"],
                    summary_text=row["summary_text"],
                    summary_type=row["summary_type"],
                    chunk_count=row["chunk_count"],
                    total_tokens=row["total_tokens"],
                    processing_time=row["processing_time"],
                    created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
                    updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
                    model_used=row["model_used"],
                    temperature=row["temperature"],
                    max_tokens=row["max_tokens"],
                    is_favorite=bool(row["is_favorite"]),
                    tags=json.loads(row["tags"]),
                    metadata=json.loads(row["metadata"])
                ) for row in rows]

        except sqlite3.Error as e:
            self.logger.error(f"Failed to get summaries for document {document_id}: {e}")
            raise

    def get_all_summaries(self, limit: Optional[int] = None) -> List[Summary]:
        """Retrieve all summaries"""
        try:
            with self.get_connection() as conn:
                query = "SELECT * FROM summaries ORDER BY created_at DESC"
                if limit:
                    query += f" LIMIT {limit}"

                rows = conn.execute(query).fetchall()

                return [Summary(
                    id=row["id"],
                    document_id=row["document_id"],
                    summary_text=row["summary_text"],
                    summary_type=row["summary_type"],
                    chunk_count=row["chunk_count"],
                    total_tokens=row["total_tokens"],
                    processing_time=row["processing_time"],
                    created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
                    updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
                    model_used=row["model_used"],
                    temperature=row["temperature"],
                    max_tokens=row["max_tokens"],
                    is_favorite=bool(row["is_favorite"]),
                    tags=json.loads(row["tags"]),
                    metadata=json.loads(row["metadata"])
                ) for row in rows]

        except sqlite3.Error as e:
            self.logger.error(f"Failed to get summaries: {e}")
            raise

    def delete_summary(self, summary_id: int) -> bool:
        """Delete a summary by ID"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM summaries WHERE id = ?", (summary_id,)
                )

                success = cursor.rowcount > 0
                conn.commit()

                if success:
                    self.logger.info(f"Deleted summary {summary_id}")
                return success

        except sqlite3.Error as e:
            self.logger.error(f"Failed to delete summary {summary_id}: {e}")
            raise

    # Processing History operations
    def save_processing_history(self, history: ProcessingHistory) -> int:
        """Save processing history to the database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO processing_history (
                        document_id, status, stage, progress, error_message,
                        completed_at, processing_time, retry_count, max_retries,
                        log_entries, metadata, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    history.document_id,
                    history.status,
                    history.stage,
                    history.progress,
                    history.error_message,
                    history.completed_at,
                    history.processing_time,
                    history.retry_count,
                    history.max_retries,
                    json.dumps(history.log_entries),
                    json.dumps(history.metadata)
                ))

                history_id = cursor.lastrowid
                conn.commit()
                self.logger.info(f"Saved processing history for document {history.document_id} with ID {history_id}")
                return history_id

        except sqlite3.Error as e:
            self.logger.error(f"Failed to save processing history: {e}")
            raise

    def update_processing_history(self, history: ProcessingHistory) -> bool:
        """Update processing history"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("""
                    UPDATE processing_history SET
                        status = ?, stage = ?, progress = ?, error_message = ?,
                        completed_at = ?, processing_time = ?, retry_count = ?,
                        log_entries = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (
                    history.status,
                    history.stage,
                    history.progress,
                    history.error_message,
                    history.completed_at,
                    history.processing_time,
                    history.retry_count,
                    json.dumps(history.log_entries),
                    json.dumps(history.metadata),
                    history.id
                ))

                success = cursor.rowcount > 0
                conn.commit()

                if success:
                    self.logger.info(f"Updated processing history {history.id}")
                return success

        except sqlite3.Error as e:
            self.logger.error(f"Failed to update processing history {history.id}: {e}")
            raise

    def get_processing_history_for_document(self, document_id: int) -> List[ProcessingHistory]:
        """Retrieve processing history for a document"""
        try:
            with self.get_connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM processing_history WHERE document_id = ? ORDER BY started_at DESC",
                    (document_id,)
                ).fetchall()

                return [ProcessingHistory(
                    id=row["id"],
                    document_id=row["document_id"],
                    status=row["status"],
                    stage=row["stage"],
                    progress=row["progress"],
                    error_message=row["error_message"],
                    started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
                    completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
                    processing_time=row["processing_time"],
                    retry_count=row["retry_count"],
                    max_retries=row["max_retries"],
                    log_entries=json.loads(row["log_entries"]),
                    metadata=json.loads(row["metadata"])
                ) for row in rows]

        except sqlite3.Error as e:
            self.logger.error(f"Failed to get processing history for document {document_id}: {e}")
            raise

    # Utility methods
    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics"""
        try:
            with self.get_connection() as conn:
                stats = {}

                # Count documents
                stats["total_documents"] = conn.execute("SELECT COUNT(*) FROM pdf_documents").fetchone()[0]
                stats["processed_documents"] = conn.execute("SELECT COUNT(*) FROM pdf_documents WHERE is_processed = 1").fetchone()[0]

                # Count summaries
                stats["total_summaries"] = conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]
                stats["favorite_summaries"] = conn.execute("SELECT COUNT(*) FROM summaries WHERE is_favorite = 1").fetchone()[0]

                # Count processing history
                stats["total_processing_history"] = conn.execute("SELECT COUNT(*) FROM processing_history").fetchone()[0]
                stats["failed_processes"] = conn.execute("SELECT COUNT(*) FROM processing_history WHERE status = 'failed'").fetchone()[0]

                return stats

        except sqlite3.Error as e:
            self.logger.error(f"Failed to get database stats: {e}")
            raise

    def cleanup_old_records(self, days: int = 30) -> int:
        """Clean up old processing history records"""
        try:
            with self.get_connection() as conn:
                cutoff_date = datetime.now().replace(microsecond=0) - timedelta(days=days)

                cursor = conn.execute(
                    "DELETE FROM processing_history WHERE started_at < ? AND status IN ('completed', 'failed')",
                    (cutoff_date.isoformat(),)
                )

                deleted_count = cursor.rowcount
                conn.commit()

                self.logger.info(f"Cleaned up {deleted_count} old processing history records")
                return deleted_count

        except sqlite3.Error as e:
            self.logger.error(f"Failed to cleanup old records: {e}")
            raise


# Global database manager instance
db_manager = None


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance"""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager


def initialize_database(db_path: Optional[str] = None):
    """Initialize the database with optional custom path"""
    global db_manager
    db_manager = DatabaseManager(db_path)
    return db_manager