"""
Logging utilities for the async chat stream system.
"""
import logging
from pathlib import Path
from typing import Optional
from ..config.constants import LOG_FORMAT, LOGS_FOLDER

def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level
        
    Returns:
        logging.Logger: Configured logger instance
        
    Raises:
        ValueError: If logger name is invalid
        OSError: If unable to create log file
    """
    if not name or not isinstance(name, str):
        raise ValueError("Logger name must be a non-empty string")
        
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if path provided
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            raise OSError(f"Failed to create log file handler: {str(e)}")
            
    return logger 