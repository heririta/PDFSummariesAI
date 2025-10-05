"""
Configuration Management Module

Provides comprehensive configuration management with validation,
error handling, and type hints for the PDF Summarizer AI application.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class LogLevel(Enum):
    """Log level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SummaryType(Enum):
    """Summary type enumeration"""
    CONCISE = "concise"
    DETAILED = "detailed"
    BULLET_POINTS = "bullet_points"
    EXECUTIVE = "executive"


@dataclass
class GroqConfig:
    """Groq API configuration"""
    api_key: str
    model: str = "llama-3.3-70b-versatile"
    base_url: str = "https://api.groq.com"
    timeout: int = 30
    max_retries: int = 3

    def __post_init__(self):
        """Validate Groq configuration"""
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required")
        if self.timeout <= 0:
            raise ValueError("GROQ_TIMEOUT must be positive")
        if self.max_retries < 0:
            raise ValueError("GROQ_MAX_RETRIES must be non-negative")


@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str = "sqlite:///data/pdf_summaries.db"
    path: str = "data/pdf_summaries.db"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10

    def __post_init__(self):
        """Validate database configuration"""
        if not self.url:
            raise ValueError("DATABASE_URL is required")
        if self.pool_size <= 0:
            raise ValueError("Database pool size must be positive")
        if self.max_overflow < 0:
            raise ValueError("Database max overflow must be non-negative")


@dataclass
class FileUploadConfig:
    """File upload configuration"""
    max_file_size: int = 52428800  # 50MB in bytes
    allowed_extensions: List[str] = field(default_factory=lambda: ["pdf"])
    upload_dir: str = "uploads"
    max_concurrent_uploads: int = 5

    def __post_init__(self):
        """Validate file upload configuration"""
        if self.max_file_size <= 0:
            raise ValueError("MAX_FILE_SIZE must be positive")
        if not self.allowed_extensions:
            raise ValueError("At least one allowed extension is required")
        if self.max_concurrent_uploads <= 0:
            raise ValueError("MAX_CONCURRENT_UPLOADS must be positive")


@dataclass
class TextProcessingConfig:
    """Text processing configuration"""
    chunk_size: int = 4000
    chunk_overlap: int = 400
    max_tokens: int = 8000
    min_chunk_size: int = 100

    def __post_init__(self):
        """Validate text processing configuration"""
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("CHUNK_OVERLAP must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        if self.max_tokens <= 0:
            raise ValueError("MAX_TOKENS must be positive")
        if self.min_chunk_size <= 0:
            raise ValueError("MIN_CHUNK_SIZE must be positive")


@dataclass
class SummaryConfig:
    """Summary generation configuration"""
    default_summary_type: str = "concise"
    max_summary_length: int = 1000
    temperature: float = 0.3

    def __post_init__(self):
        """Validate summary configuration"""
        if self.default_summary_type not in [st.value for st in SummaryType]:
            raise ValueError(f"Invalid summary type: {self.default_summary_type}")
        if self.max_summary_length <= 0:
            raise ValueError("MAX_SUMMARY_LENGTH must be positive")
        if not 0 <= self.temperature <= 2:
            raise ValueError("TEMPERATURE must be between 0 and 2")


@dataclass
class AppConfig:
    """Application configuration"""
    name: str = "PDF Summarizer AI"
    version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    secret_key: str = "default-secret-key-change-in-production"
    session_timeout: int = 3600

    def __post_init__(self):
        """Validate application configuration"""
        if not self.name:
            raise ValueError("APP_NAME is required")
        if not self.version:
            raise ValueError("APP_VERSION is required")
        if self.log_level not in [ll.value for ll in LogLevel]:
            raise ValueError(f"Invalid log level: {self.log_level}")
        if self.session_timeout <= 0:
            raise ValueError("SESSION_TIMEOUT must be positive")


@dataclass
class DirectoryConfig:
    """Directory configuration"""
    data_dir: str = "data"
    logs_dir: str = "logs"
    exports_dir: str = "exports"
    static_dir: str = "static"
    templates_dir: str = "templates"

    def __post_init__(self):
        """Validate directory configuration"""
        for attr_name, attr_value in self.__dict__.items():
            if not attr_value:
                raise ValueError(f"{attr_name.upper()} is required")


@dataclass
class StreamlitConfig:
    """Streamlit configuration"""
    page_title: str = "PDF Summarizer AI"
    page_icon: str = "📄"
    layout: str = "centered"
    wide_mode: bool = False

    def __post_init__(self):
        """Validate Streamlit configuration"""
        if not self.page_title:
            raise ValueError("PAGE_TITLE is required")
        if self.layout not in ["centered", "wide"]:
            raise ValueError("LAYOUT must be 'centered' or 'wide'")


@dataclass
class PerformanceConfig:
    """Performance configuration"""
    max_history_entries: int = 100
    cleanup_interval: int = 3600  # 1 hour in seconds
    enable_profiling: bool = False

    def __post_init__(self):
        """Validate performance configuration"""
        if self.max_history_entries <= 0:
            raise ValueError("MAX_HISTORY_ENTRIES must be positive")
        if self.cleanup_interval <= 0:
            raise ValueError("CLEANUP_INTERVAL must be positive")


class Config:
    """
    Main configuration class that manages all application settings.

    This class provides a centralized way to access all configuration values
    with proper validation and error handling.
    """

    def __init__(self):
        """Initialize configuration from environment variables"""
        self._load_configurations()
        self._validate_configurations()
        self._setup_logging()

    def _load_configurations(self):
        """Load all configuration sections"""
        self.groq = GroqConfig(
            api_key=os.getenv("GROQ_API_KEY", ""),
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com"),
            timeout=int(os.getenv("GROQ_TIMEOUT", "30")),
            max_retries=int(os.getenv("GROQ_MAX_RETRIES", "3"))
        )

        self.database = DatabaseConfig(
            url=os.getenv("DATABASE_URL", "sqlite:///data/pdf_summaries.db"),
            path=os.getenv("DATABASE_PATH", "data/pdf_summaries.db"),
            echo=os.getenv("DATABASE_ECHO", "false").lower() == "true",
            pool_size=int(os.getenv("DATABASE_POOL_SIZE", "5")),
            max_overflow=int(os.getenv("DATABASE_MAX_OVERFLOW", "10"))
        )

        self.file_upload = FileUploadConfig(
            max_file_size=int(os.getenv("MAX_FILE_SIZE", "52428800")),
            allowed_extensions=os.getenv("ALLOWED_EXTENSIONS", "pdf").split(","),
            upload_dir=os.getenv("UPLOAD_DIR", "uploads"),
            max_concurrent_uploads=int(os.getenv("MAX_CONCURRENT_UPLOADS", "5"))
        )

        self.text_processing = TextProcessingConfig(
            chunk_size=int(os.getenv("CHUNK_SIZE", "4000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "400")),
            max_tokens=int(os.getenv("MAX_TOKENS", "8000")),
            min_chunk_size=int(os.getenv("MIN_CHUNK_SIZE", "100"))
        )

        self.summary = SummaryConfig(
            default_summary_type=os.getenv("DEFAULT_SUMMARY_TYPE", "concise"),
            max_summary_length=int(os.getenv("MAX_SUMMARY_LENGTH", "1000")),
            temperature=float(os.getenv("TEMPERATURE", "0.3"))
        )

        self.app = AppConfig(
            name=os.getenv("APP_NAME", "PDF Summarizer AI"),
            version=os.getenv("APP_VERSION", "1.0.0"),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            secret_key=os.getenv("SECRET_KEY", "default-secret-key-change-in-production"),
            session_timeout=int(os.getenv("SESSION_TIMEOUT", "3600"))
        )

        self.directories = DirectoryConfig(
            data_dir=os.getenv("DATA_DIR", "data"),
            logs_dir=os.getenv("LOGS_DIR", "logs"),
            exports_dir=os.getenv("EXPORTS_DIR", "exports"),
            static_dir=os.getenv("STATIC_DIR", "static"),
            templates_dir=os.getenv("TEMPLATES_DIR", "templates")
        )

        self.streamlit = StreamlitConfig(
            page_title=os.getenv("PAGE_TITLE", "PDF Summarizer AI"),
            page_icon=os.getenv("PAGE_ICON", "📄"),
            layout=os.getenv("LAYOUT", "centered"),
            wide_mode=os.getenv("WIDE_MODE", "false").lower() == "true"
        )

        self.performance = PerformanceConfig(
            max_history_entries=int(os.getenv("MAX_HISTORY_ENTRIES", "100")),
            cleanup_interval=int(os.getenv("CLEANUP_INTERVAL", "3600")),
            enable_profiling=os.getenv("ENABLE_PROFILING", "false").lower() == "true"
        )

    def _validate_configurations(self):
        """Validate all configuration sections"""
        try:
            # Check for required configurations
            if not self.groq.api_key:
                raise ValueError("GROQ_API_KEY is required. Please set it in your .env file.")

            # Ensure directories exist
            for dir_name in ["data", "logs", "exports", "static", "templates", "uploads"]:
                dir_path = Path(os.getenv(f"{dir_name.upper()}_DIR", dir_name))
                dir_path.mkdir(parents=True, exist_ok=True)

        except Exception as e:
            print(f"Configuration validation failed: {e}", file=sys.stderr)
            sys.exit(1)

    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.app.log_level.upper())

        # Create logs directory if it doesn't exist
        log_dir = Path(self.directories.logs_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "app.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def get_database_url(self) -> str:
        """Get database URL with proper path handling"""
        return self.database.url

    def get_upload_path(self, filename: str) -> str:
        """Get full path for uploaded file"""
        upload_dir = Path(self.file_upload.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        return str(upload_dir / filename)

    def get_export_path(self, filename: str) -> str:
        """Get full path for exported file"""
        export_dir = Path(self.directories.exports_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        return str(export_dir / filename)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "groq": self.groq.__dict__,
            "database": self.database.__dict__,
            "file_upload": self.file_upload.__dict__,
            "text_processing": self.text_processing.__dict__,
            "summary": self.summary.__dict__,
            "app": self.app.__dict__,
            "directories": self.directories.__dict__,
            "streamlit": self.streamlit.__dict__,
            "performance": self.performance.__dict__
        }

    def __str__(self) -> str:
        """String representation of configuration"""
        return f"Config(app_name={self.app.name}, version={self.app.version}, debug={self.app.debug})"


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance"""
    return config


def reload_config():
    """Reload configuration from environment variables"""
    global config
    config = Config()


# Convenience functions for backward compatibility
def get_groq_api_key() -> str:
    """Get Groq API key"""
    return config.groq.api_key


def get_database_url() -> str:
    """Get database URL"""
    return config.get_database_url()


def get_max_file_size() -> int:
    """Get maximum file size"""
    return config.file_upload.max_file_size


def get_chunk_size() -> int:
    """Get chunk size"""
    return config.text_processing.chunk_size


def get_chunk_overlap() -> int:
    """Get chunk overlap"""
    return config.text_processing.chunk_overlap


if __name__ == "__main__":
    # Test configuration loading
    try:
        cfg = get_config()
        print(f"Configuration loaded successfully: {cfg}")
        print(f"Database URL: {cfg.get_database_url()}")
        print(f"Max file size: {cfg.get_max_file_size()} bytes")
        print(f"Chunk size: {cfg.get_chunk_size()}")
        print(f"Chunk overlap: {cfg.get_chunk_overlap()}")
    except Exception as e:
        print(f"Configuration error: {e}")
        sys.exit(1)