"""
Database Initialization Module

Provides database initialization functions and utilities for the PDF Summarizer AI application.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from .models import DatabaseManager, get_db_manager, initialize_database
from ..utils.config import get_config

logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """
    Database initializer class that handles database setup, migrations, and maintenance.
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize database initializer"""
        config = get_config()
        self.db_path = db_path or config.database.path
        self.logger = logging.getLogger(__name__)

    def initialize_full_database(self) -> DatabaseManager:
        """
        Initialize the complete database with all tables, indexes, and default data.

        Returns:
            DatabaseManager instance
        """
        try:
            self.logger.info("Starting database initialization...")

            # Initialize database manager
            db_manager = initialize_database(self.db_path)

            # Run migrations if needed
            self._run_migrations(db_manager)

            # Create default data if needed
            self._create_default_data(db_manager)

            # Validate database integrity
            self._validate_database(db_manager)

            self.logger.info("Database initialization completed successfully")
            return db_manager

        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise

    def _run_migrations(self, db_manager: DatabaseManager):
        """Run database migrations"""
        try:
            # Create migration table if it doesn't exist
            with db_manager.get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS migrations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        migration_name TEXT UNIQUE NOT NULL,
                        executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Check for existing migrations
                executed_migrations = {
                    row[0] for row in conn.execute("SELECT migration_name FROM migrations").fetchall()
                }

                # Run pending migrations
                migrations = self._get_migrations()

                for migration in migrations:
                    if migration["name"] not in executed_migrations:
                        self.logger.info(f"Running migration: {migration['name']}")
                        migration["function"](conn)
                        conn.execute(
                            "INSERT INTO migrations (migration_name) VALUES (?)",
                            (migration["name"],)
                        )
                        conn.commit()
                        self.logger.info(f"Migration {migration['name']} completed")

        except sqlite3.Error as e:
            self.logger.error(f"Failed to run migrations: {e}")
            raise

    def _get_migrations(self) -> list:
        """Get list of database migrations"""
        return [
            {
                "name": "add_document_indexes",
                "function": self._migration_add_document_indexes
            },
            {
                "name": "add_summary_tags_support",
                "function": self._migration_add_summary_tags_support
            },
            {
                "name": "add_processing_performance_fields",
                "function": self._migration_add_processing_performance_fields
            },
        ]

    def _migration_add_document_indexes(self, conn: sqlite3.Connection):
        """Add performance indexes for documents table"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_documents_filename ON pdf_documents(filename)",
            "CREATE INDEX IF NOT EXISTS idx_documents_upload_date ON pdf_documents(upload_date)",
            "CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON pdf_documents(file_hash)",
            "CREATE INDEX IF NOT EXISTS idx_documents_is_processed ON pdf_documents(is_processed)",
        ]
        for index_sql in indexes:
            conn.execute(index_sql)

    def _migration_add_summary_tags_support(self, conn: sqlite3.Connection):
        """Add tags support to summaries table"""
        # Check if tags column exists
        cursor = conn.execute("PRAGMA table_info(summaries)")
        columns = [row[1] for row in cursor.fetchall()]

        if "tags" not in columns:
            conn.execute("ALTER TABLE summaries ADD COLUMN tags TEXT DEFAULT '[]'")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_summaries_tags ON summaries(tags)")

    def _migration_add_processing_performance_fields(self, conn: sqlite3.Connection):
        """Add performance tracking fields to processing_history"""
        # Check if performance fields exist
        cursor = conn.execute("PRAGMA table_info(processing_history)")
        columns = [row[1] for row in cursor.fetchall()]

        fields_to_add = {
            "processing_time": "REAL DEFAULT 0.0",
            "retry_count": "INTEGER DEFAULT 0",
            "max_retries": "INTEGER DEFAULT 3",
            "log_entries": "TEXT DEFAULT '[]'"
        }

        for field, field_type in fields_to_add.items():
            if field not in columns:
                conn.execute(f"ALTER TABLE processing_history ADD COLUMN {field} {field_type}")

    def _create_default_data(self, db_manager: DatabaseManager):
        """Create default data if needed"""
        try:
            with db_manager.get_connection() as conn:
                # Create application settings table if it doesn't exist
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS app_settings (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Insert default settings
                default_settings = {
                    "app_version": "1.0.0",
                    "max_file_size": str(get_config().file_upload.max_file_size),
                    "default_summary_type": get_config().summary.default_summary_type,
                    "chunk_size": str(get_config().text_processing.chunk_size),
                    "chunk_overlap": str(get_config().text_processing.chunk_overlap),
                    "auto_cleanup_days": "30",
                    "enable_analytics": "true"
                }

                for key, value in default_settings.items():
                    conn.execute("""
                        INSERT OR IGNORE INTO app_settings (key, value)
                        VALUES (?, ?)
                    """, (key, value))

                conn.commit()
                self.logger.info("Default data created successfully")

        except sqlite3.Error as e:
            self.logger.error(f"Failed to create default data: {e}")
            raise

    def _validate_database(self, db_manager: DatabaseManager):
        """Validate database integrity"""
        try:
            # Check if all required tables exist
            required_tables = {"pdf_documents", "summaries", "processing_history", "migrations", "app_settings"}

            with db_manager.get_connection() as conn:
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                existing_tables = {row[0] for row in cursor.fetchall()}

                missing_tables = required_tables - existing_tables
                if missing_tables:
                    raise ValueError(f"Missing required tables: {missing_tables}")

                # Check foreign key constraints
                cursor = conn.execute("PRAGMA foreign_key_check")
                violations = cursor.fetchall()

                if violations:
                    self.logger.warning(f"Found {len(violations)} foreign key violations")
                    for violation in violations:
                        self.logger.warning(f"FK Violation: {violation}")

                # Run integrity check
                cursor = conn.execute("PRAGMA integrity_check")
                integrity_result = cursor.fetchone()[0]

                if integrity_result != "ok":
                    raise ValueError(f"Database integrity check failed: {integrity_result}")

            self.logger.info("Database validation completed successfully")

        except Exception as e:
            self.logger.error(f"Database validation failed: {e}")
            raise

    def backup_database(self, backup_path: Optional[str] = None) -> str:
        """
        Create a backup of the database.

        Args:
            backup_path: Optional path for backup file

        Returns:
            Path to the backup file
        """
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"backup_{timestamp}.db"

            source_db = Path(self.db_path)
            backup_db = Path(backup_path)

            # Ensure backup directory exists
            backup_db.parent.mkdir(parents=True, exist_ok=True)

            # Create backup
            with sqlite3.connect(str(source_db)) as source:
                with sqlite3.connect(str(backup_db)) as backup:
                    source.backup(backup)

            self.logger.info(f"Database backup created: {backup_path}")
            return str(backup_db)

        except Exception as e:
            self.logger.error(f"Failed to create database backup: {e}")
            raise

    def restore_database(self, backup_path: str) -> bool:
        """
        Restore database from backup.

        Args:
            backup_path: Path to backup file

        Returns:
            True if restore successful
        """
        try:
            backup_db = Path(backup_path)

            if not backup_db.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_path}")

            # Validate backup file
            with sqlite3.connect(str(backup_db)) as backup:
                cursor = backup.execute("PRAGMA integrity_check")
                integrity_result = cursor.fetchone()[0]

                if integrity_result != "ok":
                    raise ValueError(f"Backup file integrity check failed: {integrity_result}")

            # Create current database backup before restore
            current_backup = self.backup_database()

            # Restore from backup
            with sqlite3.connect(str(backup_db)) as backup:
                with sqlite3.connect(self.db_path) as target:
                    backup.backup(target)

            self.logger.info(f"Database restored from: {backup_path}")
            self.logger.info(f"Previous database backed up to: {current_backup}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to restore database: {e}")
            raise

    def reset_database(self, confirm: bool = False) -> bool:
        """
        Reset database by deleting and recreating all tables.

        Args:
            confirm: Confirmation flag to prevent accidental resets

        Returns:
            True if reset successful
        """
        if not confirm:
            raise ValueError("Database reset requires explicit confirmation")

        try:
            self.logger.warning("Starting database reset...")

            # Create backup before reset
            backup_path = self.backup_database()
            self.logger.info(f"Database backed up to: {backup_path}")

            # Delete and recreate database
            db_file = Path(self.db_path)
            if db_file.exists():
                db_file.unlink()

            # Initialize fresh database
            db_manager = initialize_database(self.db_path)
            self._create_default_data(db_manager)

            self.logger.info("Database reset completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Database reset failed: {e}")
            raise

    def get_database_info(self) -> Dict[str, Any]:
        """Get comprehensive database information"""
        try:
            db_manager = get_db_manager()
            stats = db_manager.get_database_stats()

            with db_manager.get_connection() as conn:
                # Get table information
                cursor = conn.execute("""
                    SELECT name, sql FROM sqlite_master
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                    ORDER BY name
                """)
                tables = {row[0]: row[1] for row in cursor.fetchall()}

                # Get index information
                cursor = conn.execute("""
                    SELECT name, tbl_name, sql FROM sqlite_master
                    WHERE type='index' AND name NOT LIKE 'sqlite_%'
                    ORDER BY tbl_name, name
                """)
                indexes = [
                    {
                        "name": row[0],
                        "table": row[1],
                        "sql": row[2]
                    }
                    for row in cursor.fetchall()
                ]

                # Get database file size
                db_file = Path(self.db_path)
                file_size = db_file.stat().st_size if db_file.exists() else 0

                # Get SQLite version
                cursor = conn.execute("SELECT sqlite_version()")
                sqlite_version = cursor.fetchone()[0]

            return {
                "database_path": self.db_path,
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "sqlite_version": sqlite_version,
                "statistics": stats,
                "tables": tables,
                "indexes": indexes,
                "last_backup": self._get_last_backup_info()
            }

        except Exception as e:
            self.logger.error(f"Failed to get database info: {e}")
            raise

    def _get_last_backup_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the last backup"""
        try:
            backup_pattern = "backup_*.db"
            backup_files = list(Path().glob(backup_pattern))

            if not backup_files:
                return None

            # Get most recent backup
            latest_backup = max(backup_files, key=lambda p: p.stat().st_mtime)
            backup_time = datetime.fromtimestamp(latest_backup.stat().st_mtime)
            backup_size = latest_backup.stat().st_size

            return {
                "file": str(latest_backup),
                "created_at": backup_time.isoformat(),
                "size_bytes": backup_size,
                "size_mb": round(backup_size / (1024 * 1024), 2)
            }

        except Exception as e:
            self.logger.error(f"Failed to get backup info: {e}")
            return None


def setup_database() -> DatabaseManager:
    """
    Setup and initialize the database.

    Returns:
        Initialized DatabaseManager instance
    """
    initializer = DatabaseInitializer()
    return initializer.initialize_full_database()


def backup_database(backup_path: Optional[str] = None) -> str:
    """
    Create a database backup.

    Args:
        backup_path: Optional backup file path

    Returns:
        Path to the backup file
    """
    initializer = DatabaseInitializer()
    return initializer.backup_database(backup_path)


def restore_database(backup_path: str) -> bool:
    """
    Restore database from backup.

    Args:
        backup_path: Path to backup file

    Returns:
        True if restore successful
    """
    initializer = DatabaseInitializer()
    return initializer.restore_database(backup_path)


def get_database_info() -> Dict[str, Any]:
    """Get comprehensive database information"""
    initializer = DatabaseInitializer()
    return initializer.get_database_info()


if __name__ == "__main__":
    # Test database initialization
    try:
        print("Initializing database...")
        db_manager = setup_database()
        print("Database initialized successfully")

        print("\nDatabase information:")
        info = get_database_info()
        print(f"Database path: {info['database_path']}")
        print(f"File size: {info['file_size_mb']} MB")
        print(f"SQLite version: {info['sqlite_version']}")
        print(f"Statistics: {info['statistics']}")

        print("\nDatabase setup completed successfully!")

    except Exception as e:
        print(f"Database setup failed: {e}")
        exit(1)