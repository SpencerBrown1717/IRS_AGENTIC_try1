"""
Logging configuration and utilities for IRS data agent.
"""
import os
import logging
import sys
from logging.handlers import RotatingFileHandler
from threading import Lock
from pathlib import Path
from typing import Optional, Dict, Any

# Global configuration
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_FILE = "logs/irs_data_agent.log"
DEFAULT_MAX_SIZE_MB = 10
DEFAULT_BACKUP_COUNT = 5

# Thread safety for logger creation
_logger_lock = Lock()
_loggers = {}

def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
    max_size_mb: int = DEFAULT_MAX_SIZE_MB,
    backup_count: int = DEFAULT_BACKUP_COUNT
) -> None:
    """
    Configure global logging settings.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        log_format: Log message format
        console_output: Whether to output logs to console
        file_output: Whether to output logs to file
        max_size_mb: Maximum log file size in MB before rotation
        backup_count: Number of backup log files to keep
    """
    # Load configuration
    try:
        from irs_data_agent.utils.config import load_config
        config = load_config()
        logging_config = config.get("logging", {})
    except Exception:
        logging_config = {}
    
    # Set parameters from arguments, config, or defaults
    level_str = log_level or logging_config.get("level", DEFAULT_LOG_LEVEL)
    log_file = log_file or logging_config.get("file", DEFAULT_LOG_FILE)
    log_format = log_format or logging_config.get("format", DEFAULT_LOG_FORMAT)
    
    # Fix: Use the provided values if not None, otherwise use config or default
    max_size_mb = max_size_mb if max_size_mb is not None else logging_config.get("max_size_mb", DEFAULT_MAX_SIZE_MB)
    backup_count = backup_count if backup_count is not None else logging_config.get("backup_count", DEFAULT_BACKUP_COUNT)
    
    # Convert level string to logging level
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    level = level_map.get(level_str.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        root_logger.addHandler(console_handler)
    
    # Add file handler if requested
    if file_output and log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Create rotating file handler
        max_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
    
    # Log configuration details
    root_logger.info(f"Logging configured: level={level_str}, file={log_file if file_output else 'disabled'}")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    global _loggers
    
    # Thread safety for logger creation
    with _logger_lock:
        if name not in _loggers:
            logger = logging.getLogger(name)
            _loggers[name] = logger
        
        return _loggers[name]

def log_exception(logger: logging.Logger, exc: Exception, context: Dict[str, Any] = None) -> None:
    """
    Log an exception with additional context.
    
    Args:
        logger: Logger instance
        exc: Exception to log
        context: Additional context to include in log
    """
    error_message = f"Exception: {type(exc).__name__}: {str(exc)}"
    
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        error_message += f" | Context: {context_str}"
    
    logger.error(error_message, exc_info=True)

def log_api_call(
    logger: logging.Logger,
    method: str,
    endpoint: str,
    status_code: int = None,
    duration_ms: int = None
) -> None:
    """
    Log an API call with performance metrics.
    
    Args:
        logger: Logger instance
        method: HTTP method
        endpoint: API endpoint
        status_code: Response status code
        duration_ms: Request duration in milliseconds
    """
    message = f"API {method} {endpoint}"
    
    if status_code is not None:
        message += f" | Status: {status_code}"
    
    if duration_ms is not None:
        message += f" | Duration: {duration_ms}ms"
    
    log_level = logging.INFO if status_code is None or status_code < 400 else logging.ERROR
    logger.log(log_level, message)

def get_log_level() -> str:
    """
    Get the current root logger level.
    
    Returns:
        Log level name
    """
    level = logging.getLogger().getEffectiveLevel()
    
    level_names = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: "WARNING",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "CRITICAL"
    }
    
    return level_names.get(level, "UNKNOWN")
