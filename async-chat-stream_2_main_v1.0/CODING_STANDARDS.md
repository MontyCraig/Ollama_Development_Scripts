# Python Coding Standards

## Overview

This document outlines the coding standards and best practices for production-ready Python code in this project.

## 1. Type Safety

### Type Hints

```python
from typing import List, Dict, Optional, Union, Any, TypeVar, Generic

def process_items(items: List[str]) -> Dict[str, int]:
    """Process string items and return count mapping."""
    return {item: len(item) for item in items}

class DataProcessor[T]:  # Python 3.12+ generic syntax
    def process(self, data: T) -> Optional[T]:
        """Process generic data type."""
        return data if self.validate(data) else None
```

### Type Checking

* Use mypy for static type checking
* Enable strict mode in mypy.ini
* No implicit Any types
* Validate return types

## 2. Error Handling

### Exception Hierarchy

```python
class AppError(Exception):
    """Base exception for application."""
    pass

class ValidationError(AppError):
    """Validation-specific errors."""
    pass

class APIError(AppError):
    """API-related errors."""
    pass
```

### Try-Except Patterns

```python
try:
    result = potentially_dangerous_operation()
except ValidationError as e:
    logger.error(f"Validation failed: {str(e)}")
    raise
except Exception as e:
    logger.critical(f"Unexpected error: {str(e)}")
    raise AppError(f"Operation failed: {str(e)}") from e
else:
    logger.info("Operation completed successfully")
    return result
finally:
    cleanup_resources()
```

## 3. Documentation

### Docstring Format

```python
def complex_operation(
    param1: str,
    param2: Optional[int] = None
) -> Dict[str, Any]:
    """Perform a complex operation with detailed documentation.

    Args:
        param1: Description of param1
        param2: Description of param2, defaults to None

    Returns:
        Dict containing operation results

    Raises:
        ValidationError: If params are invalid
        APIError: If external API fails

    Example:
        >>> result = complex_operation("test", 42)
        >>> print(result)
        {'status': 'success', 'value': 42}
    """
    pass
```

## 4. Input Validation

### Parameter Validation

```python
from dataclasses import dataclass
from pydantic import BaseModel, Field

@dataclass(frozen=True)
class ValidatedInput:
    """Immutable input container with validation."""
    value: str
    
    def __post_init__(self):
        if not self.value.strip():
            raise ValidationError("Value cannot be empty")

class APIRequest(BaseModel):
    """Pydantic model for API request validation."""
    id: int = Field(gt=0)
    name: str = Field(min_length=1)
```

### Sanitization

```python
import html
import re
from typing import Any

def sanitize_input(value: Any) -> str:
    """Sanitize user input for safe processing.
    
    Args:
        value: Raw input value
        
    Returns:
        Sanitized string
    """
    # Convert to string
    str_value = str(value)
    
    # Remove control characters
    str_value = "".join(char for char in str_value if ord(char) >= 32)
    
    # Escape HTML
    str_value = html.escape(str_value)
    
    # Remove potential injection patterns
    str_value = re.sub(r'[\'";{}()]+', '', str_value)
    
    return str_value.strip()
```

## 5. Logging

### Setup

```python
import logging
from logging.handlers import RotatingFileHandler
import sys

def setup_logger(
    name: str,
    log_file: str,
    level: int = logging.INFO
) -> logging.Logger:
    """Configure logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        '%(levelname)s: %(message)s'
    ))
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
```

### Usage

```python
logger = setup_logger(__name__, "app.log")

def process_data(data: Dict[str, Any]) -> None:
    """Process data with comprehensive logging."""
    logger.info(f"Starting data processing: {len(data)} items")
    
    try:
        logger.debug(f"Raw data: {data}")
        
        # Validate
        if not data:
            logger.warning("Empty data received")
            return
            
        # Process
        for key, value in data.items():
            logger.info(f"Processing item: {key}")
            try:
                result = transform_item(value)
                logger.debug(f"Transformed {key}: {result}")
            except Exception as e:
                logger.error(f"Failed to process {key}: {str(e)}")
                continue
                
    except Exception as e:
        logger.critical(f"Critical error in processing: {str(e)}")
        raise
    else:
        logger.info("Data processing completed successfully")
```

## 6. Testing

### Test Structure

```python
import pytest
from unittest.mock import Mock, patch

def test_complex_operation():
    """Test complex operation with various scenarios."""
    # Arrange
    test_data = {"key": "value"}
    mock_api = Mock()
    
    # Act
    with patch("module.api_client", mock_api):
        result = complex_operation(test_data)
    
    # Assert
    assert result["status"] == "success"
    mock_api.call.assert_called_once()

@pytest.mark.parametrize("input,expected", [
    ("valid", True),
    ("", False),
    (None, False)
])
def test_validation(input: Optional[str], expected: bool):
    """Parameterized test for input validation."""
    assert validate_input(input) == expected
```

## 7. Code Organization

### Module Structure

```
module/
├── __init__.py
├── core.py
├── exceptions.py
├── types.py
├── utils/
│   ├── __init__.py
│   ├── validation.py
│   └── helpers.py
└── tests/
    ├── __init__.py
    ├── test_core.py
    └── test_utils.py
```

### Import Order

```python
# Standard library
import os
import sys
from typing import Optional

# Third-party packages
import pandas as pd
import numpy as np

# Local modules
from .core import process_data
from .exceptions import ValidationError
from .utils.helpers import sanitize_input
```

## 8. Performance Considerations

### Resource Management

```python
from contextlib import contextmanager

@contextmanager
def resource_manager():
    """Manage resource lifecycle."""
    logger.info("Acquiring resource")
    resource = acquire_resource()
    try:
        yield resource
    finally:
        logger.info("Releasing resource")
        resource.release()

def process_with_resource():
    """Process using managed resource."""
    with resource_manager() as resource:
        return resource.process()
```

### Optimization

* Use generators for large datasets
* Implement caching where appropriate
* Profile code for bottlenecks
* Use appropriate data structures

## 9. Security

### Best Practices

* No hardcoded secrets
* Use environment variables
* Implement rate limiting
* Validate all external input
* Use secure dependencies
* Regular security updates

## 10. Maintenance

### Code Review Checklist

* Type safety verified
* Tests added/updated
* Documentation complete
* Error handling comprehensive
* Logging appropriate
* Security considered
* Performance acceptable

### Versioning

* Semantic versioning
* Changelog maintained
* Breaking changes documented

## 11. Data Validation and Type Safety

### Pydantic vs. Traditional Type Hints

#### Pydantic Benefits

* **Runtime Validation**: Pydantic performs actual runtime validation, not just static type checking
* **Data Parsing**: Automatically converts input data to declared types
* **Schema Generation**: Auto-generates JSON Schema for models
* **Serialization**: Built-in JSON serialization/deserialization
* **Nested Models**: Elegant handling of complex nested data structures
* **Built-in Validators**: Rich set of validators and custom validation decorators
* **IDE Support**: Excellent autocomplete and type inference
* **Testing**: Includes `pydantic.BaseModel.model_validate()` for easy testing
* **Performance**: Compiled with Cython for high-performance validation

Example Pydantic Model:

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1)
    tokens: Optional[int] = None
    
    class Config:
        frozen = True  # Immutable after creation

class ChatSession(BaseModel):
    messages: List[ChatMessage]
    model: str
    temperature: float = Field(default=0.7, ge=0, le=2.0)
```

#### Traditional Type Hints

```python
from typing import List, Optional, TypedDict

class ChatMessage(TypedDict):
    role: str
    content: str
    tokens: Optional[int]

# Requires manual validation
def validate_message(msg: ChatMessage) -> bool:
    if msg["role"] not in {"user", "assistant", "system"}:
        return False
    if not msg["content"]:
        return False
    return True
```

### When to Use Each Approach

#### Use Pydantic When:

* Handling external data (API requests, config files)
* Need runtime validation
* Working with complex data structures
* Building APIs or data models
* Need automatic serialization
* Want comprehensive validation rules

#### Use Traditional Type Hints When:

* Simple internal data structures
* Static type checking is sufficient
* Minimal runtime overhead is crucial
* No need for data validation/parsing
* Limited to Python 3.6+ compatibility

### Testing Considerations

#### Pydantic Testing Features

```python
from pydantic import BaseModel, ValidationError
import pytest

def test_chat_message():
    # Valid message
    msg = ChatMessage(role="user", content="Hello")
    assert msg.role == "user"
    
    # Invalid role
    with pytest.raises(ValidationError):
        ChatMessage(role="invalid", content="Hello")
        
    # Invalid content
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="")
```

#### Integration with pytest

* Use `pytest.mark.parametrize` with Pydantic models
* Leverage `model_validate_json()` for JSON testing
* Use `model_dump()` for serialization testing

### Performance Considerations

* Pydantic v2+ is significantly faster than v1
* Use `frozen=True` for immutable models
* Consider using `Config.extra = 'ignore'` to skip validation of extra fields
* Cache validated models when possible


