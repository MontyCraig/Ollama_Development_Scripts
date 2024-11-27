#!/usr/bin/env python3
"""
Validate Ollama model configuration files.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfigValidator:
    """Validates Ollama model configuration files."""
    
    def __init__(self, config_dir: str = '.'):
        self.config_dir = Path(config_dir)
        self.required_files = [
            'ollama_models_details.json',
            'ollama_embeddings_models.json',
            'ollama_tool_use_models.json',
            'ollama_vision_models.json',
            'ollama_available_models.csv'
        ]
        
    def validate_json_file(self, filepath: Path) -> bool:
        """Validate JSON file structure and content."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            logger.info(f"Successfully validated {filepath.name}")
            return True
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {filepath}: {str(e)}")
            return False
            
    def validate_csv_file(self, filepath: Path) -> bool:
        """Validate CSV file structure."""
        try:
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                list(reader)  # Validate entire file
            logger.info(f"Successfully validated {filepath.name}")
            return True
        except Exception as e:
            logger.error(f"Invalid CSV in {filepath}: {str(e)}")
            return False
            
    def validate_all(self) -> bool:
        """Validate all configuration files."""
        all_valid = True
        
        for filename in self.required_files:
            filepath = self.config_dir / filename
            if not filepath.exists():
                logger.error(f"Missing required file: {filename}")
                all_valid = False
                continue
                
            if filename.endswith('.json'):
                if not self.validate_json_file(filepath):
                    all_valid = False
            elif filename.endswith('.csv'):
                if not self.validate_csv_file(filepath):
                    all_valid = False
                    
        return all_valid

    def validate_tool_schema(self, schema: Dict) -> bool:
        """Validate tool function schema."""
        required_fields = {
            'type': str,
            'function': dict
        }
        
        function_fields = {
            'name': str,
            'description': str,
            'parameters': dict
        }
        
        try:
            # Validate top level
            for field, field_type in required_fields.items():
                if field not in schema:
                    logger.error(f"Missing required field: {field}")
                    return False
                if not isinstance(schema[field], field_type):
                    logger.error(f"Invalid type for {field}")
                    return False
                    
            # Validate function object
            func = schema['function']
            for field, field_type in function_fields.items():
                if field not in func:
                    logger.error(f"Missing function field: {field}")
                    return False
                if not isinstance(func[field], field_type):
                    logger.error(f"Invalid type for function.{field}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Schema validation error: {str(e)}")
            return False

if __name__ == '__main__':
    validator = ConfigValidator()
    if validator.validate_all():
        logger.info("All configuration files are valid")
    else:
        logger.error("Some configuration files are invalid") 