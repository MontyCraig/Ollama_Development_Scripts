"""
File handling utilities for the async chat stream system.
"""
import os
from pathlib import Path
from typing import Union, Optional
import re
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def sanitize_path(path: Union[str, Path]) -> Path:
    """
    Sanitize and validate file path.
    
    Args:
        path: Input path to sanitize
        
    Returns:
        Path: Sanitized absolute path
        
    Raises:
        ValueError: If path is invalid or contains directory traversal
    """
    if not path:
        raise ValueError("Path cannot be empty")
        
    try:
        # Convert to Path and resolve
        clean_path = Path(path).resolve()
        
        # Check for directory traversal
        if ".." in str(clean_path):
            raise ValueError("Directory traversal detected")
            
        logger.debug(f"Sanitized path: {clean_path}")
        return clean_path
        
    except Exception as e:
        logger.error(f"Path sanitization failed: {str(e)}")
        raise ValueError(f"Invalid path: {str(e)}")

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists and create if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path: Created/existing directory path
        
    Raises:
        OSError: If directory creation fails
    """
    try:
        dir_path = sanitize_path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        os.chmod(dir_path, 0o755)
        logger.info(f"Ensured directory exists: {dir_path}")
        return dir_path
    except Exception as e:
        logger.error(f"Failed to create directory: {str(e)}")
        raise OSError(f"Failed to create directory: {str(e)}")

def get_unique_filename(
    base_path: Union[str, Path],
    extension: str,
    prefix: Optional[str] = None
) -> Path:
    """
    Generate unique filename with optional prefix.
    
    Args:
        base_path: Base directory path
        extension: File extension
        prefix: Optional filename prefix
        
    Returns:
        Path: Unique file path
        
    Raises:
        ValueError: If inputs are invalid
    """
    try:
        base_path = sanitize_path(base_path)
        if not extension.startswith('.'):
            extension = f".{extension}"
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}" if prefix else timestamp
        
        # Sanitize filename
        filename = re.sub(r'[^\w\s-]', '', filename)
        full_path = base_path / f"{filename}{extension}"
        
        counter = 1
        while full_path.exists():
            new_name = f"{filename}_{counter}{extension}"
            full_path = base_path / new_name
            counter += 1
            if counter > 1000:
                raise ValueError("Too many file versions")
                
        logger.debug(f"Generated unique filename: {full_path}")
        return full_path
        
    except Exception as e:
        logger.error(f"Failed to generate unique filename: {str(e)}")
        raise 