"""Logging utilities for the Mean Reversion Trading System."""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger

from .config import get_config


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    rotation: Optional[str] = None,
    retention: Optional[str] = None
) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file
        log_format: Log message format
        rotation: Log rotation policy
        retention: Log retention policy
    """
    config = get_config()
    
    # Get configuration values with defaults
    log_level = log_level or config.get('logging.level', 'INFO')
    log_file = log_file or config.get('logging.file', 'logs/trading_system.log')
    log_format = log_format or config.get(
        'logging.format', 
        "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}"
    )
    rotation = rotation or config.get('logging.rotation', '1 week')
    retention = retention or config.get('logging.retention', '1 month')
    
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stdout,
        level=log_level,
        format=log_format,
        colorize=True
    )
    
    # Add file handler if log_file is specified
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            level=log_level,
            format=log_format,
            rotation=rotation,
            retention=retention,
            compression="zip"
        )


def get_logger(name: str) -> "logger":
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    # Setup logging if not already done
    if not logger._core.handlers:
        setup_logging()
    
    return logger.bind(name=name)


# Setup logging on module import
setup_logging()
