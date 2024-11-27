# Chat Module

## Overview
The Chat module demonstrates professional-grade integration with Ollama for local LLM deployment. This module serves as an educational resource for implementing secure, efficient, and type-safe chat interactions with local language models.

## Ollama Integration Examples

### 1. Basic Chat Session
```python
from typing import AsyncGenerator
from .chat_manager import ChatManager
from .types import Message, Response

async def basic_chat_example() -> None:
    """Demonstrate basic Ollama chat integration."""
    async with ChatManager() as chat:
        # Initialize chat with model
        await chat.initialize_model("llama2")
        
        # Send message and stream response
        async for chunk in chat.send_message(
            "Explain Python async/await"
        ):
            print(chunk.content, end="", flush=True)
```

### 2. Advanced Chat Features
```python
from typing import List
from .chat_manager import ChatManager
from .types import ChatMode, ModelConfig

async def advanced_chat_example() -> None:
    """Demonstrate advanced Ollama features."""
    config = ModelConfig(
        name="codellama",
        context_length=4096,
        temperature=0.7
    )
    
    async with ChatManager(config) as chat:
        # Enable function calling
        chat.enable_function_calling()
        
        # Set chat mode
        chat.set_mode(ChatMode.CREATIVE)
        
        # Get response with metadata
        response = await chat.get_response_with_meta(
            "Create a FastAPI app"
        )
        
        print(f"Response: {response.content}")
        print(f"Tokens: {response.token_count}")
        print(f"Model: {response.model_name}")
```

### 3. Chain of Thought
```python
from .chat_manager import ChatManager
from .prompts import ChainOfThoughtPrompt

async def chain_of_thought_example() -> None:
    """Demonstrate chain of thought prompting."""
    async with ChatManager() as chat:
        prompt = ChainOfThoughtPrompt(
            task="Design a database schema",
            steps=[
                "Identify entities",
                "Define relationships",
                "Specify constraints",
                "Create SQL statements"
            ]
        )
        
        response = await chat.execute_chain(prompt)
        print(response.reasoning)
        print(response.solution)
```

## Type Safety Examples

### 1. Message Types
```python
from pydantic import BaseModel
from typing import Optional, List, Union
from enum import Enum

class Role(str, Enum):
    """Chat message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class Message(BaseModel):
    """Type-safe chat message."""
    role: Role
    content: str
    metadata: Optional[dict] = None
    
    @validator("content")
    def validate_content(cls, v: str) -> str:
        """Validate message content."""
        v = v.strip()
        if not v:
            raise ValueError("Empty message")
        return v
```

### 2. Response Handling
```python
from typing import AsyncGenerator, TypeVar
from .types import Response, StreamChunk

T = TypeVar("T", bound=Response)

async def process_stream(
    stream: AsyncGenerator[StreamChunk, None],
    handler: Callable[[StreamChunk], T]
) -> List[T]:
    """Process response stream with type safety."""
    responses: List[T] = []
    async for chunk in stream:
        processed = handler(chunk)
        responses.append(processed)
    return responses
```

## Error Handling

### 1. Custom Exceptions
```python
class OllamaError(Exception):
    """Base exception for Ollama operations."""
    pass

class ModelNotFoundError(OllamaError):
    """Model not found in Ollama."""
    pass

class ChatError(OllamaError):
    """Chat operation error."""
    def __init__(
        self,
        message: str,
        context: Optional[dict] = None
    ):
        super().__init__(message)
        self.context = context or {}
```

### 2. Error Recovery
```python
async def safe_chat_operation(
    operation: Callable,
    max_retries: int = 3
) -> Response:
    """Safe chat operation with retries."""
    for attempt in range(max_retries):
        try:
            return await operation()
        except ModelNotFoundError:
            raise  # Don't retry if model missing
        except ChatError as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(
                f"Retry {attempt + 1}/{max_retries}: {e}"
            )
            await asyncio.sleep(1)
```

## Logging Integration

### 1. Structured Logging
```python
import logging
from typing import Any, Dict

class ChatLogger:
    """Chat-specific logger."""
    
    def __init__(self) -> None:
        self.logger = logging.getLogger("chat")
        
    def log_interaction(
        self,
        message: Message,
        response: Response,
        metadata: Dict[str, Any]
    ) -> None:
        """Log chat interaction with context."""
        self.logger.info(
            "Chat interaction",
            extra={
                "message": message.dict(),
                "response": response.dict(),
                "metadata": metadata
            }
        )
```

## Testing Strategy

### 1. Unit Tests
```python
import pytest
from unittest.mock import AsyncMock
from .chat_manager import ChatManager

@pytest.mark.asyncio
async def test_chat_session():
    """Test chat session lifecycle."""
    manager = ChatManager()
    
    async with manager as chat:
        response = await chat.send_message(
            "Test message"
        )
        assert response is not None
        assert response.content
```

### 2. Integration Tests
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_ollama_integration():
    """Test actual Ollama integration."""
    async with ChatManager() as chat:
        await chat.initialize_model("llama2")
        response = await chat.send_message(
            "Simple test"
        )
        assert response.model_name == "llama2"
```

## Security Considerations

### 1. Input Validation
```python
from pydantic import BaseModel, validator
from typing import Optional

class ChatInput(BaseModel):
    """Validated chat input."""
    
    message: str
    system_prompt: Optional[str] = None
    
    @validator("message")
    def validate_message(cls, v: str) -> str:
        """Validate chat message."""
        v = v.strip()
        if not v:
            raise ValueError("Empty message")
        if len(v) > 4096:
            raise ValueError("Message too long")
        return v
```

### 2. Resource Management
```python
from contextlib import asynccontextmanager
from typing import AsyncGenerator

@asynccontextmanager
async def managed_chat_session(
    model: str,
    timeout: float = 30.0
) -> AsyncGenerator[ChatManager, None]:
    """Manage chat session with timeouts."""
    manager = ChatManager()
    try:
        await asyncio.wait_for(
            manager.initialize(),
            timeout=timeout
        )
        yield manager
    finally:
        await manager.cleanup()
```

## Performance Optimization

### 1. Caching
```python
from functools import lru_cache
from typing import Optional

class ResponseCache:
    """Cache for common responses."""
    
    @lru_cache(maxsize=1000)
    def get_cached_response(
        self,
        message: str,
        context: str
    ) -> Optional[Response]:
        """Get cached response if available."""
        key = self._create_cache_key(message, context)
        return self._cache.get(key)
```

### 2. Batch Processing
```python
async def process_message_batch(
    messages: List[Message],
    batch_size: int = 10
) -> List[Response]:
    """Process messages in batches."""
    responses = []
    for batch in chunks(messages, batch_size):
        batch_responses = await asyncio.gather(
            *(process_message(msg) for msg in batch)
        )
        responses.extend(batch_responses)
    return responses
```

## Best Practices

### 1. Code Organization
- Separate message handling
- Clean interfaces
- Resource management
- Error recovery

### 2. Documentation
- Clear examples
- Type information
- Error scenarios
- Usage patterns

### 3. Testing
- Unit test coverage
- Integration tests
- Error case testing
- Performance testing

### 4. Security
- Input validation
- Resource limits
- Error handling
- Logging security
