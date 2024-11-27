"""
Custom type definitions for the async chat stream system.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

class ModelSize(Enum):
    """Enumeration of model size categories."""
    EMBEDDING = "embedding"  # Embedding models
    SMALL = "small"         # < 5GB
    MEDIUM = "medium"       # 5GB - 20GB
    LARGE = "large"        # > 20GB

@dataclass
class OllamaModel:
    """Data class representing an Ollama model."""
    name: str
    size: float  # in GB
    size_category: ModelSize
    is_embedding: bool = False

@dataclass
class ChatMessage:
    """Data class representing a chat message."""
    role: str
    content: str
    timestamp: str

@dataclass
class ChatSession:
    """Data class representing a chat session."""
    id: str
    name: Optional[str]
    messages: List[ChatMessage]
    model_name: str
    created_at: str
    updated_at: str 