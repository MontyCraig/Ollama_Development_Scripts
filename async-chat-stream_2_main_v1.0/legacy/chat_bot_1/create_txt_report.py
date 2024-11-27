#!/usr/bin/env python3
"""
JSON Chat Report Generator
-------------------------
This script converts JSON chat files into readable text reports, including chat content,
model information, and metadata. It processes only new JSON files that don't have 
corresponding reports yet.

Author: Assistant
Version: 1.1
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Union, Set
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChatReportGenerator:
    """Handles the conversion of JSON chat files to formatted text reports."""
    
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize the ChatReportGenerator.
        
        Args:
            input_dir (str): Directory containing JSON chat files
            output_dir (str): Directory where text reports will be saved
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def get_existing_reports(self) -> Set[str]:
        """
        Get a set of JSON filenames that already have corresponding reports.
        
        Returns:
            Set[str]: Set of JSON filenames (without extension) that have reports
        """
        existing_reports = set()
        for report_file in self.output_dir.glob('*_report.txt'):
            # Remove '_report' suffix and get original JSON filename
            json_name = report_file.stem.rsplit('_report', 1)[0]
            existing_reports.add(json_name)
        return existing_reports

    def needs_processing(self, json_file: Path, existing_reports: Set[str]) -> bool:
        """
        Check if a JSON file needs to be processed.
        
        Args:
            json_file (Path): Path to JSON file
            existing_reports (Set[str]): Set of already processed JSON filenames
            
        Returns:
            bool: True if file needs processing, False otherwise
        """
        # Check if report exists
        if json_file.stem in existing_reports:
            report_file = self.output_dir / f"{json_file.stem}_report.txt"
            
            # If report exists, check if JSON is newer
            if report_file.exists():
                json_mtime = json_file.stat().st_mtime
                report_mtime = report_file.stat().st_mtime
                
                # Only process if JSON is newer than report
                return json_mtime > report_mtime
                
        return True

    def load_json_file(self, file_path: Path) -> Dict:
        """
        Load and parse a JSON file with enhanced error handling.
        
        Args:
            file_path (Path): Path to the JSON file
            
        Returns:
            Dict: Parsed JSON content
            
        Raises:
            json.JSONDecodeError: If JSON parsing fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                logger.debug(f"Raw content from {file_path.name}: {content[:100]}...")  # Log first 100 chars
                return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in {file_path.name} at position {e.pos}: {e.msg}")
            raise
        except UnicodeDecodeError as e:
            logger.error(f"Unicode decode error in {file_path.name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error reading {file_path.name}: {e}")
            raise
            
    def format_chat_message(self, message: Dict) -> str:
        """
        Format a single chat message.
        
        Args:
            message (Dict): Message dictionary containing role and content
            
        Returns:
            str: Formatted message string
        """
        role = message.get('role', 'unknown').upper()
        content = message.get('content', '')
        return f"\n[{role}]:\n{content}\n"
    
    def generate_report_content(self, data: Union[Dict, List], filename: str) -> str:
        """
        Generate the complete report content from chat data.
        
        Args:
            data (Union[Dict, List]): Parsed JSON chat data, either a dict or list
            filename (str): Original JSON filename
            
        Returns:
            str: Formatted report content
        """
        report = [
            "=" * 80,
            f"Chat Report - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
            f"\nSource File: {filename}\n"
        ]
        
        # Handle both dictionary and list formats
        if isinstance(data, dict):
            # Add model information if available
            if 'model' in data:
                report.append(f"Model: {data['model']}\n")
            messages = data.get('messages', [])
        else:
            # If data is a list, treat it directly as messages
            messages = data
            
        report.append("Chat History:")
        report.append("-" * 80)
        
        for message in messages:
            report.append(self.format_chat_message(message))
            
        report.append("-" * 80)
        return "\n".join(report)
    
    def process_single_file(self, json_file: Path) -> None:
        """
        Process a single JSON file and generate/update its report.
        
        Args:
            json_file (Path): Path to the JSON file to process
        """
        try:
            logger.info(f"Processing single file: {json_file.name}")
            
            existing_reports = self.get_existing_reports()
            
            if not self.needs_processing(json_file, existing_reports):
                logger.info(f"Skipping {json_file.name} - report already exists and is up to date")
                return
                
            data = self.load_json_file(json_file)
            
            # Generate report content
            report_content = self.generate_report_content(data, json_file.name)
            
            # Create output file
            output_file = self.output_dir / f"{json_file.stem}_report.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
                
            logger.info(f"Generated/Updated report: {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing {json_file.name}: {str(e)}")
            logger.exception("Detailed error information:")

    def process_files(self) -> None:
        """
        Process all JSON files in the input directory and generate text reports.
        """
        json_files = list(self.input_dir.glob('*.json'))
        
        if not json_files:
            logger.warning(f"No JSON files found in {self.input_dir}")
            return
            
        for json_file in json_files:
            self.process_single_file(json_file)

def main():
    """Main function to run the report generator."""
    try:
        # Get the current directory where the script is located
        current_dir = Path(__file__).parent
        
        # Update paths relative to chat_bot_1 directory
        chats_dir = current_dir / 'chats'
        reports_dir = current_dir / 'reports'
        
        # Enable debug logging for this run
        logger.setLevel(logging.DEBUG)
        
        logger.info(f"Processing files from: {chats_dir}")
        logger.info(f"Saving reports to: {reports_dir}")
        
        generator = ChatReportGenerator(
            input_dir=str(chats_dir),
            output_dir=str(reports_dir)
        )
        generator.process_files()
        
    except Exception as e:
        logger.error(f"Error running report generator: {e}")
        logger.exception("Detailed error information:")
        raise

if __name__ == "__main__":
    main()
