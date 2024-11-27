#!/usr/bin/env python3
"""
Cleanup script for dynamic directories.
WARNING: Currently disabled - keeping test files.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import List, Optional

class DirectoryCleaner:
    """Manages cleanup of dynamic directories."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.dynamic_dirs = [
            'task_lists',
            'output_folders',
            'model_tests',
            'reports'
        ]
        # Directories to preserve for now
        self.preserve_dirs = [
            'chats',
            'conversations',
            'convos',
            'json_output',
            'json_outputs'
        ]
        
    def cleanup_directory(self, 
                         dir_path: Path,
                         retain_days: Optional[int] = None,
                         dry_run: bool = True) -> None:
        """
        Clean up a directory while preserving structure.
        
        Args:
            dir_path: Directory to clean
            retain_days: Optional days of data to keep
            dry_run: If True, only log what would be done
        """
        if not dir_path.exists():
            return
            
        if dir_path.name in self.preserve_dirs:
            logging.info(f"Preserving test directory: {dir_path}")
            return
            
        for item in dir_path.iterdir():
            if item.name == '.gitkeep':
                continue
                
            if retain_days:
                mtime = datetime.fromtimestamp(item.stat().st_mtime)
                if datetime.now() - mtime < timedelta(days=retain_days):
                    continue
                    
            if dry_run:
                logging.info(f"Would remove: {item}")
            else:
                if item.is_file():
                    item.unlink()
                else:
                    shutil.rmtree(item)
                logging.info(f"Removed: {item}")
                
    def cleanup_all(self,
                   retain_days: Optional[int] = None,
                   dry_run: bool = True) -> None:
        """
        Clean up all dynamic directories.
        
        Args:
            retain_days: Optional days of data to keep
            dry_run: If True, only log what would be done
        """
        for dir_name in self.dynamic_dirs:
            dir_path = self.base_dir / dir_name
            self.cleanup_directory(dir_path, retain_days, dry_run)

if __name__ == '__main__':
    # Currently disabled - uncomment when ready to use
    """
    logging.basicConfig(level=logging.INFO)
    base_dir = Path(__file__).parent.parent
    cleaner = DirectoryCleaner(base_dir)
    
    # Dry run by default
    cleaner.cleanup_all(retain_days=30, dry_run=True)
    """
    print("Cleanup script is currently disabled to preserve test files.") 