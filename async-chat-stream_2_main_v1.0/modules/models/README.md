# Models Module

This module contains the core data models for the Async Chat Stream Processing System, implemented using Pydantic.

## Data Models Overview

### Core Models
- `ChatMessage`: Immutable message model with role validation
- `ChatSession`: Session management with message history
- `OllamaModel`: Ollama model metadata and configuration
- `Task`: Task tracking and state management

### Enums
- `MessageRole`: Valid message roles (user/assistant/system)
- `TaskStatus`: Task state tracking

### Helper Models
- `ModelSize`: Size tracking with byte/GB validation

## Features

### Type Safety
- Full type hints with runtime validation
- Immutable models where appropriate
- Enum-based constraints

### Validation Rules
- Message content length validation
- Temperature range constraints (0-2.0)
- Timestamp consistency checks
- Size conversion validation

### Metadata Support
- Flexible metadata fields
- UUID-based identification
- Timestamp tracking

## Usage Examples

### Creating a Chat Message
```python
from models.data_models import ChatMessage, MessageRole

message = ChatMessage(
    role=MessageRole.USER,
    content="Hello, how can you help me?",
    metadata={"client": "web"}
)
```

### Managing a Chat Session
```python
from models.data_models import ChatSession

session = ChatSession(
    model="llama2",
    temperature=0.7
)
session.messages.append(message)
```

### Working with Tasks
```python
from models.data_models import Task, TaskStatus

task = Task(
    name="Model Download",
    description="Download llama2 model",
    status=TaskStatus.PENDING
)
```

## Validation Examples

### Temperature Validation
```python
# This will raise ValidationError
session = ChatSession(
    model="llama2",
    temperature=3.0  # Must be between 0 and 2.0
)
```

### Message Role Validation
```python
# This will raise ValidationError
message = ChatMessage(
    role="invalid_role",  # Must be user/assistant/system
    content="Hello"
)
```

## Testing

### Unit Tests
The models include comprehensive validation that can be tested:
```python
import pytest
from models.data_models import ChatMessage, MessageRole

def test_chat_message_validation():
    with pytest.raises(ValidationError):
        ChatMessage(role="invalid", content="")
```

### Integration Tests
Test the models with actual Ollama responses:
```python
async def test_ollama_integration():
    response = await client.chat(...)
    message = ChatMessage(
        role=MessageRole.ASSISTANT,
        content=response.content
    )
```

## Best Practices

1. Always use Enums for constrained fields
2. Leverage metadata for extensibility
3. Use frozen=True for immutable data
4. Include examples in Config.json_schema_extra
5. Add docstrings for all models
6. Implement custom validators when needed

## Performance Considerations

- Use `frozen=True` for immutable models
- Consider using `Config.extra = 'ignore'` for parsing
- Validate only when necessary
- Cache validated models when possible

## Future Enhancements

1. Add more specific validation rules
2. Implement model serialization formats
3. Add database integration
4. Expand metadata schemas
5. Add more helper methods

## Dependencies

- pydantic >= 2.6.1
- python >= 3.8

## Ollama Model Management

### 1. Model Categories
```python
from enum import Enum
from typing import Dict, List
from pydantic import BaseModel

class ModelSize(str, Enum):
    """Model size categories."""
    SMALL = "small"    # < 5GB
    MEDIUM = "medium"  # 5-20GB
    LARGE = "large"    # > 20GB

class ModelInfo(BaseModel):
    """Model information container."""
    name: str
    size: ModelSize
    parameters: int
    context_length: int
    quantization: str
    
class ModelManager:
    """Ollama model manager."""
    
    async def get_models(
        self,
        size: ModelSize = None
    ) -> List[ModelInfo]:
        """Get available models."""
        models = await self._fetch_models()
        if size:
            return [m for m in models if m.size == size]
        return models
```

### 2. Model Operations
```python
from typing import Optional
from .exceptions import ModelError

class ModelOperations:
    """Model operation handlers."""
    
    async def download_model(
        self,
        name: str,
        quantization: str = "q4_0"
    ) -> ModelInfo:
        """Download Ollama model."""
        try:
            info = await self._get_model_info(name)
            if not self._check_disk_space(info.size_bytes):
                raise ModelError("Insufficient disk space")
            
            model = await self._download(
                name,
                quantization=quantization
            )
            return model
            
        except Exception as e:
            raise ModelError(f"Download failed: {e}")
    
    async def remove_model(
        self,
        name: str
    ) -> None:
        """Remove Ollama model."""
        try:
            await self._remove(name)
        except Exception as e:
            raise ModelError(f"Removal failed: {e}")
```

### 3. Model Selection
```python
from typing import List, Optional
from .types import ModelCriteria, ModelInfo

class ModelSelector:
    """Intelligent model selection."""
    
    async def select_model(
        self,
        criteria: ModelCriteria
    ) -> ModelInfo:
        """Select best model for criteria."""
        models = await self.get_models()
        
        # Filter by requirements
        candidates = [
            m for m in models
            if self._meets_criteria(m, criteria)
        ]
        
        if not candidates:
            raise ModelError("No suitable model found")
            
        # Rank by performance/size trade-off
        ranked = self._rank_models(candidates)
        return ranked[0]
```

## Type Safety Examples

### 1. Model Types
```python
from pydantic import BaseModel, validator
from typing import Optional, Dict, Any

class ModelConfig(BaseModel):
    """Model configuration."""
    name: str
    context_length: int
    temperature: float = 0.7
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    
    @validator("temperature")
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Invalid temperature")
        return v
```

### 2. Model Statistics
```python
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ModelStats:
    """Model usage statistics."""
    total_tokens: int
    average_latency: float
    error_rate: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "tokens": self.total_tokens,
            "latency": self.average_latency,
            "errors": self.error_rate
        }
```

## Error Handling

### 1. Custom Exceptions
```python
class ModelError(Exception):
    """Base model exception."""
    pass

class ModelNotFoundError(ModelError):
    """Model not found in Ollama."""
    pass

class ModelDownloadError(ModelError):
    """Model download failure."""
    def __init__(
        self,
        model: str,
        reason: str,
        details: Optional[Dict] = None
    ):
        super().__init__(f"{model}: {reason}")
        self.details = details or {}
```

### 2. Error Recovery
```python
from typing import Optional
import asyncio

async def safe_model_operation(
    operation: Callable,
    max_retries: int = 3,
    backoff: float = 1.0
) -> Any:
    """Execute model operation with retries."""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            return await operation()
        except ModelNotFoundError:
            raise  # Don't retry if model missing
        except ModelError as e:
            last_error = e
            if attempt < max_retries - 1:
                await asyncio.sleep(backoff * (attempt + 1))
                
    raise last_error
```

## Logging Integration

### 1. Model Events
```python
import logging
from typing import Dict, Any

class ModelLogger:
    """Model-specific logger."""
    
    def __init__(self) -> None:
        self.logger = logging.getLogger("models")
        
    def log_model_event(
        self,
        event: str,
        model: str,
        details: Dict[str, Any]
    ) -> None:
        """Log model event with context."""
        self.logger.info(
            f"Model event: {event}",
            extra={
                "model": model,
                "event": event,
                "details": details
            }
        )
```

## Testing Strategy

### 1. Unit Tests
```python
import pytest
from unittest.mock import AsyncMock
from .model_manager import ModelManager

@pytest.mark.asyncio
async def test_model_download():
    """Test model download process."""
    manager = ModelManager()
    
    model = await manager.download_model(
        "llama2",
        quantization="q4_0"
    )
    
    assert model.name == "llama2"
    assert model.quantization == "q4_0"
```

### 2. Integration Tests
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_model_operations():
    """Test full model lifecycle."""
    manager = ModelManager()
    
    # Download
    model = await manager.download_model("llama2")
    assert await manager.model_exists("llama2")
    
    # Use
    info = await manager.get_model_info("llama2")
    assert info.name == "llama2"
    
    # Remove
    await manager.remove_model("llama2")
    assert not await manager.model_exists("llama2")
```

## Security Considerations

### 1. Resource Management
```python
from typing import Dict
import psutil

class ResourceMonitor:
    """Monitor system resources."""
    
    def check_resources(
        self,
        required: Dict[str, float]
    ) -> bool:
        """Check if system can handle model."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return (
            memory.available >= required["memory"]
            and disk.free >= required["disk"]
        )
```

### 2. Model Validation
```python
from hashlib import sha256
from typing import Optional

class ModelValidator:
    """Validate model integrity."""
    
    async def validate_model(
        self,
        name: str,
        expected_hash: Optional[str] = None
    ) -> bool:
        """Validate downloaded model."""
        model_path = self._get_model_path(name)
        if not model_path.exists():
            return False
            
        if expected_hash:
            actual_hash = sha256(
                model_path.read_bytes()
            ).hexdigest()
            return actual_hash == expected_hash
            
        return True
```

## Performance Optimization

### 1. Model Caching
```python
from functools import lru_cache
from typing import Dict, Optional

class ModelCache:
    """Cache for model metadata."""
    
    @lru_cache(maxsize=100)
    def get_model_metadata(
        self,
        name: str
    ) -> Optional[Dict]:
        """Get cached model metadata."""
        return self._metadata.get(name)
```

### 2. Parallel Operations
```python
async def batch_model_operation(
    models: List[str],
    operation: Callable,
    max_concurrent: int = 3
) -> List[Any]:
    """Execute operations in parallel."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def bounded_operation(model: str) -> Any:
        async with semaphore:
            return await operation(model)
            
    return await asyncio.gather(
        *(bounded_operation(m) for m in models)
    )
```

## Best Practices

### 1. Code Organization
- Model categorization
- Operation isolation
- Resource management
- Error handling

### 2. Documentation
- Clear examples
- Type information
- Error scenarios
- Usage patterns

### 3. Testing
- Unit test coverage
- Integration tests
- Resource testing
- Error case testing

### 4. Security
- Resource limits
- Model validation
- Access control
- Error handling
