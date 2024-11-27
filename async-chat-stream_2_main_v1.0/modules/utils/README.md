# Utils Module

## Overview

The Utils module provides essential utility functions for the Ollama integration system, demonstrating professional-grade Python development practices. This module serves as an educational resource for implementing robust, type-safe, and secure utility functions.

## Educational Value

### 1. Type Safety Examples

```python
from typing import Dict, List, Optional, Union
from pathlib import Path

def safe_file_read(
    file_path: Union[str, Path],
    encoding: str = "utf-8"
) -> Optional[str]:
    """
    Safely read file contents with proper type hints.
    
    Args:
        file_path: Path to file
        encoding: File encoding
        
    Returns:
        File contents or None if error
    """
    try:
        path = Path(file_path)
        return path.read_text(encoding=encoding)
    except Exception as e:
        logger.error(f"File read error: {e}")
        return None
```

### 2. Error Handling Patterns

```python
from typing import Any
from .exceptions import ResourceError

class ResourceManager:
    """Example of proper resource management."""
    
    async def __aenter__(self) -> "ResourceManager":
        try:
            await self.initialize()
            return self
        except Exception as e:
            raise ResourceError(f"Init failed: {e}")
            
    async def __aexit__(self, *args: Any) -> None:
        await self.cleanup()
```

### 3. Input Validation

```python
from pydantic import BaseModel, validator
from typing import Optional

class UserInput(BaseModel):
    """Example of input validation."""
    
    prompt: str
    model_name: Optional[str] = None
    
    @validator("prompt")
    def validate_prompt(cls, v: str) -> str:
        """Validate prompt content."""
        v = v.strip()
        if not v:
            raise ValueError("Empty prompt")
        return v
```

### 4. Logging Best Practices

```python
import logging
from typing import Dict, Any

def setup_logging(
    level: int = logging.INFO,
    config: Dict[str, Any] = None
) -> None:
    """Configure structured logging."""
    
    format_str = (
        "%(asctime)s - %(name)s - %(levelname)s - "
        "%(filename)s:%(lineno)d - %(message)s"
    )
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("app.log")
        ]
    )
```

## Security Considerations

### 1. Path Sanitization

```python
from pathlib import Path
import os

def safe_path_join(*parts: str) -> Path:
    """Safely join path components."""
    path = Path(*parts).resolve()
    base = Path(os.getcwd()).resolve()
    
    if not str(path).startswith(str(base)):
        raise ValueError("Path traversal detected")
    return path
```

### 2. Resource Management

```python
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Any

@asynccontextmanager
async def managed_resource(
    resource_id: str
) -> AsyncGenerator[Any, None]:
    """Safe resource management example."""
    try:
        resource = await acquire_resource(resource_id)
        yield resource
    finally:
        await release_resource(resource_id)
```

## Testing Strategy

### 1. Unit Tests

```python
import pytest
from .file_utils import safe_file_read

def test_safe_file_read_success():
    """Test successful file read."""
    content = safe_file_read("test.txt")
    assert content is not None
    
def test_safe_file_read_failure():
    """Test file read failure."""
    content = safe_file_read("nonexistent.txt")
    assert content is None
```

### 2. Integration Tests

```python
@pytest.mark.asyncio
async def test_resource_lifecycle():
    """Test complete resource lifecycle."""
    async with managed_resource("test") as resource:
        assert resource is not None
        # Test resource operations
```

## Production Code Standards

### 1. Type Safety

* Use type hints consistently
* Enable mypy strict mode
* Document type assumptions
* Handle Optional types properly

### 2. Error Handling

* Use custom exceptions
* Provide context in errors
* Clean up resources
* Log errors appropriately

### 3. Performance

* Use async where beneficial
* Implement caching
* Optimize file operations
* Profile critical paths

### 4. Documentation

* Clear docstrings
* Usage examples
* Type information
* Error scenarios

## Common Utilities

### 1. File Operations

* Safe file reading/writing
* Path manipulation
* Directory operations
* File validation

### 2. Data Processing

* JSON handling
* CSV processing
* Data validation
* Type conversion

### 3. Async Utilities

* Resource management
* Async context managers
* Task coordination
* Error handling

### 4. Security

* Input sanitization
* Path validation
* Resource limits
* Access control

## Best Practices

### 1. Code Organization

* Single responsibility
* Clear interfaces
* Proper encapsulation
* Consistent naming

### 2. Error Management

* Detailed messages
* Proper logging
* Resource cleanup
* Error recovery

### 3. Testing

* Unit test coverage
* Integration tests
* Property testing
* Error case testing

### 4. Documentation

* Clear examples
* Type information
* Error scenarios
* Usage patterns


