# Configuration Module

## Overview

The configuration module demonstrates best practices for managing system-wide constants and configuration values in a Python application.

## Educational Value

### 1. Configuration Management

* Centralized constant definitions
* Type-safe configuration values
* Environment-aware settings

### 2. Python Best Practices

* Constants naming conventions
* Type annotations
* Documentation standards

### 3. File Organization

* Separation of configuration from logic
* Module-level documentation
* Import management

## Module Structure

### constants.py

* System-wide constants
* Folder and file paths
* File extensions
* Default prompts and settings

## Usage Example

```python
from modules.config.constants import (
    LOGS_FOLDER,
    CHATS_FOLDER,
    TASKS_FOLDER,
    MODELS_FOLDER
)

# Access typed constants
log_path = Path(LOGS_FOLDER)
```

## Best Practices Demonstrated


1. Use of ALL_CAPS for constants
2. Descriptive constant names
3. Proper type annotations
4. Comprehensive docstrings
5. Logical grouping of related constants

## Constants

### Folder Structure

* `LOGS_FOLDER`: Directory for system logs
* `CHATS_FOLDER`: Directory for chat history
* `TASKS_FOLDER`: Directory for task lists
* `MODELS_FOLDER`: Directory for model configurations

### File Extensions

* `CHAT_FILE_EXT`: Extension for chat files (.json)
* `TASK_FILE_EXT`: Extension for task files (.csv)
* `LOG_FILE_EXT`: Extension for log files (.log)

### System Prompts

* `PRE_PROMPT`: System prompt prefix for AI interactions
* `POST_PROMPT`: System prompt suffix for AI interactions

### System Configuration

* `LOG_FORMAT`: Standardized logging format string
* `OLLAMA_API_HOST`: Ollama API endpoint URL

## Best Practices

### Adding New Constants


1. Use UPPERCASE for constant names
2. Add appropriate type hints
3. Document the constant's purpose
4. Update this README
5. Group related constants together

### Using Constants


1. Always import specific constants rather than using wildcards
2. Use type hints when working with constants
3. Validate paths and URLs before use
4. Handle missing or invalid configurations gracefully

### Modifying Constants


1. Consider backward compatibility
2. Update all dependent modules
3. Document changes in version control
4. Test system thoroughly after changes

## Security Considerations

* Keep sensitive information in environment variables
* Validate all paths for directory traversal
* Sanitize user inputs before combining with constants
* Use secure protocols for API endpoints

## Error Handling

When using constants:


1. Check for None/empty values
2. Validate paths exist
3. Verify file permissions
4. Handle missing configurations
5. Log configuration errors

## Maintenance

Regular maintenance tasks:


1. Review and update constants
2. Verify path configurations
3. Update documentation
4. Check for deprecated settings
5. Maintain backwards compatibility


