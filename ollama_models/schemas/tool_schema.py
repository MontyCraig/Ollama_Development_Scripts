"""
Schema definitions for Ollama tool functions.
"""

from typing import Dict, List, Union, Optional
from pydantic import BaseModel, Field

class ParameterProperty(BaseModel):
    type: str
    description: str

class FunctionParameter(BaseModel):
    type: str = "object"
    required: List[str]
    properties: Dict[str, ParameterProperty]

class Function(BaseModel):
    name: str
    description: str
    parameters: FunctionParameter

class ToolDefinition(BaseModel):
    type: str = "function"
    function: Function 