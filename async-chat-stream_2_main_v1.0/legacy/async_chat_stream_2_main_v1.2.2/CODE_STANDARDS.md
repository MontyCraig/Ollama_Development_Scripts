# Code Standards for Chat Bot Project

## General Principles




1. **Strong Typing**
   * Use type hints for all function parameters and return values
   * Use TypeVar for generic types
   * Define custom types/enums where appropriate
   * Use collections.abc for container types
2. **Documentation**
   * Every module must have a module-level docstring
   * Every function/class must have detailed docstrings
   * Use Google-style docstring format
   * Include examples in docstrings where helpful
   * Document exceptions that may be raised
3. **Error Handling**
   * Use specific exception types
   * Handle all expected error cases
   * Log all errors with appropriate context
   * Provide user-friendly error messages
   * Use custom exceptions for domain-specific errors
4. **Input Validation & Sanitization**
   * Validate all user inputs before processing
   * Sanitize inputs to prevent injection attacks
   * Check data types and ranges
   * Validate file paths and permissions
   * Use input length limits
5. **Logging**
   * Use hierarchical logging levels appropriately
   * Include contextual information in log messages
   * Log both successful operations and failures
   * Include timestamps and function names
   * Maintain separate logs for different concerns

## Project-Specific Standards




1. **Path Management**
   * Use pathlib.Path for all path operations
   * All paths must be relative to project root
   * Validate path existence before operations
   * Handle path permissions appropriately
2. **Configuration**
   * No hardcoded values in code
   * Use configuration files for constants
   * Environment-specific configs in .env files
   * Validate all config values on load
3. **Model Interaction**
   * Validate model availability before use
   * Handle model timeouts gracefully
   * Implement rate limiting
   * Cache model responses where appropriate
4. **File Operations**
   * Use context managers for file operations
   * Implement file locking where needed
   * Handle partial writes/reads
   * Validate file contents before processing

## Implementation Examples

### Strong Typing Example

```python
from typing import List, Dict, Optional, Union
from pathlib import Path

def process_chat_history(
    file_path: Path,
    max_entries: Optional[int] = None
) -> List[Dict[str, Union[str, int]]]:
    """Process chat history with proper typing."""
```

### Error Handling Example

```python
class ChatBotError(Exception):
    """Base exception for ChatBot errors."""
    pass

class ModelNotFoundError(ChatBotError):
    """Raised when a requested model is not available."""
    pass

try:
    model = get_model(model_name)
except ModelNotFoundError as e:
    logger.error(f"Model {model_name} not found: {str(e)}")
    raise
```

### Input Validation Example

```python
def validate_user_input(text: str, max_length: int = 1000) -> str:
    """
    Validate and sanitize user input.
    
    Args:
        text: Raw input text
        max_length: Maximum allowed length
        
    Returns:
        Sanitized text
        
    Raises:
        ValueError: If input is invalid
    """
    if not text or text.isspace():
        raise ValueError("Input cannot be empty")
    
    # Remove dangerous characters
    sanitized = re.sub(r'[^\w\s.,!?-]', '', text)
    
    # Enforce length limit
    if len(sanitized) > max_length:
        raise ValueError(f"Input exceeds maximum length of {max_length}")
        
    return sanitized.strip()
```

### Logging Example

```python
import logging
from functools import wraps
from typing import Callable, Any

def log_function_call(func: Callable) -> Callable:
    """Decorator to log function calls."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.info(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise
    return wrapper
```


