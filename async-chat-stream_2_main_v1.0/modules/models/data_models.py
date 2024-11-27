"""
Core data models for the Async Chat Stream Processing System.

This module defines all the Pydantic models used throughout the application for:
- Chat messages and sessions
- Ollama model configurations
- Task definitions and states
- System configurations
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, UTC
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum
import uuid


class MessageRole(str, Enum):
    """Enumeration of possible message roles in a chat."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """Represents a single message in a chat conversation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole
    content: str = Field(..., min_length=1)
    tokens: Optional[int] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class ModelSize(BaseModel):
    """Represents the size information of an Ollama model."""
    bytes: int = Field(..., gt=0)
    gigabytes: float = None  # Will be computed from bytes

    def __init__(self, **data):
        if 'bytes' in data and 'gigabytes' not in data:
            data['gigabytes'] = round(data['bytes'] / (1024 ** 3), 2)
        super().__init__(**data)

    @field_validator("gigabytes")
    @classmethod
    def validate_gigabytes(cls, v: float, info: Dict[str, Any]) -> float:
        """Ensure gigabytes matches bytes conversion."""
        if v is None:
            return 0.0
        bytes_value = info.data.get('bytes', 0)
        expected = round(bytes_value / (1024 ** 3), 2)
        if abs(v - expected) > 0.01:  # Tighter precision check
            raise ValueError(f"Gigabytes value {v} doesn't match bytes conversion {expected}")
        return expected  # Always use computed value for consistency


class OllamaModel(BaseModel):
    """Represents an Ollama model with its metadata."""
    name: str = Field(..., min_length=1)
    size: ModelSize
    is_embedding: bool = False
    parameters: Optional[Dict[str, Any]] = None
    tags: List[str] = Field(default_factory=list)
    last_used: Optional[datetime] = None


class ChatSession(BaseModel):
    """Represents a chat session with message history and settings."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[ChatMessage] = Field(default_factory=list)
    model: str
    temperature: float = Field(default=0.7, ge=0, le=2.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("updated_at")
    @classmethod
    def update_timestamp(cls, v: datetime, info: Dict[str, Any]) -> datetime:
        """Ensure updated_at is never earlier than created_at."""
        created_at = info.data.get('created_at')
        if created_at and v < created_at:
            return created_at
        return v


class TaskStatus(str, Enum):
    """Enumeration of possible task states."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task(BaseModel):
    """Represents a task in the system."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., min_length=1)
    description: Optional[str] = None
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("completed_at")
    @classmethod
    def validate_completion_time(cls, v: Optional[datetime], 
                               info: Dict[str, Any]) -> Optional[datetime]:
        """Ensure completion time is valid relative to other timestamps."""
        if v is None:
            return None
        created_at = info.data.get('created_at')
        if created_at and v < created_at:
            raise ValueError("Completion time cannot be before creation time")
        return v
