"""
Unit tests for the data models module.

This module contains comprehensive tests for all Pydantic models,
including validation rules, custom validators, and edge cases.
"""

import pytest
from datetime import datetime, timedelta, timezone
from pydantic import ValidationError
from modules.models.data_models import (
    MessageRole,
    ChatMessage,
    ModelSize,
    OllamaModel,
    ChatSession,
    TaskStatus,
    Task
)


class TestMessageRole:
    def test_valid_roles(self):
        """Test that all defined roles are valid."""
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"
        assert MessageRole.SYSTEM == "system"

    def test_invalid_role(self):
        """Test that invalid roles raise ValidationError."""
        with pytest.raises(ValidationError):
            ChatMessage(role="invalid", content="test")


class TestChatMessage:
    def test_valid_message(self):
        """Test creation of valid chat message."""
        msg = ChatMessage(
            role=MessageRole.USER,
            content="Hello, world!"
        )
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello, world!"
        assert msg.id is not None
        assert isinstance(msg.timestamp, datetime)
        assert msg.timestamp.tzinfo == timezone.utc  # Verify timezone awareness

    def test_empty_content(self):
        """Test that empty content raises ValidationError."""
        with pytest.raises(ValidationError):
            ChatMessage(role=MessageRole.USER, content="")

    def test_immutability(self):
        """Test that messages are immutable after creation."""
        msg = ChatMessage(
            role=MessageRole.USER,
            content="Hello"
        )
        with pytest.raises(Exception):  # Pydantic v2 raises different exception types
            msg.content = "Modified"


class TestModelSize:
    def test_valid_size(self):
        """Test valid model size creation with explicit values."""
        size = ModelSize(bytes=4_000_000_000)  # Only specify bytes
        assert size.bytes == 4_000_000_000
        assert size.gigabytes == 3.73  # Auto-computed and rounded

    def test_size_computation(self):
        """Test that gigabytes is correctly computed from bytes."""
        test_cases = [
            (1_073_741_824, 1.0),  # 1 GB
            (2_147_483_648, 2.0),  # 2 GB
            (5_368_709_120, 5.0),  # 5 GB
        ]
        for bytes_val, expected_gb in test_cases:
            size = ModelSize(bytes=bytes_val)
            assert size.gigabytes == expected_gb

    def test_invalid_conversion(self):
        """Test that incorrect GB conversion raises ValidationError."""
        with pytest.raises(ValidationError):
            ModelSize(
                bytes=4_000_000_000,
                gigabytes=5.0  # Incorrect conversion
            )

    def test_negative_bytes(self):
        """Test that negative bytes raises ValidationError."""
        with pytest.raises(ValidationError):
            ModelSize(bytes=-1)


class TestOllamaModel:
    def test_valid_model(self):
        """Test creation of valid Ollama model."""
        model = OllamaModel(
            name="llama2",
            size=ModelSize(bytes=4_000_000_000),  # Only specify bytes
            parameters={"context_length": 4096}
        )
        assert model.name == "llama2"
        assert model.size.gigabytes == 3.73
        assert not model.is_embedding

    def test_empty_name(self):
        """Test that empty name raises ValidationError."""
        with pytest.raises(ValidationError):
            OllamaModel(
                name="",
                size=ModelSize(bytes=4_000_000_000)
            )


class TestChatSession:
    def test_valid_session(self):
        """Test creation of valid chat session."""
        session = ChatSession(
            model="llama2",
            temperature=0.7
        )
        assert session.model == "llama2"
        assert session.temperature == 0.7
        assert len(session.messages) == 0
        assert session.created_at.tzinfo == timezone.utc
        assert session.updated_at.tzinfo == timezone.utc

    def test_invalid_temperature(self):
        """Test that invalid temperature raises ValidationError."""
        with pytest.raises(ValidationError):
            ChatSession(
                model="llama2",
                temperature=2.5  # Must be <= 2.0
            )

    def test_timestamp_validation(self):
        """Test that updated_at cannot be before created_at."""
        past = datetime.now(timezone.utc) - timedelta(days=1)
        present = datetime.now(timezone.utc)
        
        session = ChatSession(
            model="llama2",
            created_at=present,
            updated_at=past  # Should be adjusted to present
        )
        assert session.updated_at >= session.created_at


class TestTask:
    def test_valid_task(self):
        """Test creation of valid task."""
        task = Task(
            name="Download Model",
            description="Download llama2"
        )
        assert task.name == "Download Model"
        assert task.status == TaskStatus.PENDING
        assert task.created_at.tzinfo == timezone.utc
        assert task.updated_at.tzinfo == timezone.utc

    def test_completion_time_validation(self):
        """Test that completion time cannot be before creation time."""
        past = datetime.now(timezone.utc) - timedelta(days=1)
        present = datetime.now(timezone.utc)
        
        with pytest.raises(ValidationError):
            Task(
                name="Test",
                created_at=present,
                completed_at=past
            )

    def test_status_transitions(self):
        """Test valid task status transitions."""
        task = Task(name="Test")
        assert task.status == TaskStatus.PENDING
        
        task.status = TaskStatus.IN_PROGRESS
        assert task.status == TaskStatus.IN_PROGRESS
        
        task.status = TaskStatus.COMPLETED
        assert task.status == TaskStatus.COMPLETED


if __name__ == "__main__":
    pytest.main([__file__])
