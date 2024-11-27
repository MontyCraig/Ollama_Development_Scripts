"""
Model management module for Ollama chat system.
"""

import json
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from pydantic import create_model
from docstring_parser import parse

class ModelManager:
    """Manages Ollama model configurations and interactions."""
    
    def __init__(self):
        self.models_dir = Path(__file__).parent.parent.parent.parent / 'ollama_models'
        self.models_details = self._load_json('ollama_models_details.json')
        self.embeddings_models = self._load_json('ollama_embeddings_models.json')
        self.tool_use_models = self._load_json('ollama_tool_use_models.json')
        self.vision_models = self._load_json('ollama_vision_models.json')
        self.tool_functions: Dict[str, Callable] = {}
        
    def _load_json(self, filename: str) -> Dict:
        """Load JSON configuration file."""
        filepath = self.models_dir / filename
        with open(filepath, 'r') as f:
            return json.load(f)

    def register_tool(self, func: Callable) -> None:
        """
        Register a function as a tool.
        
        Args:
            func: Function to register as tool
        """
        schema = self._generate_tool_schema(func)
        self.tool_functions[func.__name__] = {
            'function': func,
            'schema': schema
        }
        
    def _generate_tool_schema(self, func: Callable) -> Dict[str, Any]:
        """
        Generate JSON schema for a function.
        
        Args:
            func: Function to generate schema for
            
        Returns:
            Dict containing function schema
        """
        sig = inspect.signature(func)
        docstring = parse(func.__doc__ or "")
        
        parameters = {
            "type": "object",
            "required": [],
            "properties": {}
        }
        
        for name, param in sig.parameters.items():
            if param.default == inspect.Parameter.empty:
                parameters["required"].append(name)
                
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else "string"
            param_doc = next((p.description for p in docstring.params if p.arg_name == name), "")
            
            parameters["properties"][name] = {
                "type": self._get_json_type(param_type),
                "description": param_doc
            }
            
        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": docstring.short_description or "",
                "parameters": parameters
            }
        }
        
    def _get_json_type(self, python_type: type) -> str:
        """Convert Python type to JSON schema type."""
        type_map = {
            int: "integer",
            float: "number",
            str: "string",
            bool: "boolean",
            list: "array",
            dict: "object"
        }
        return type_map.get(python_type, "string")
