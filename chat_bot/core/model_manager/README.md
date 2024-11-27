# Model Manager Module

Handles Ollama model discovery, validation, and lifecycle management.

## Components

### ModelManager Class

```python
class ModelManager:
    """
    Manages Ollama model discovery and lifecycle.
    
    Attributes:
        models_cache (Dict[str, ModelInfo]): Cache of discovered models
        connection_timeout (float): Timeout for model operations
        base_url (str): Ollama API endpoint
    """
```

### Key Features

1. **Model Discovery**
   * Automatic model detection
   * Model metadata caching
   * Version tracking
   * Size categorization

2. **Connection Management**
   * Health checks
   * Connection pooling
   * Timeout handling
   * Rate limiting

3. **Model Validation**
   * Availability checks
   * Version compatibility
   * Resource requirements
   * Performance metrics

### API Reference

#### Model Operations
```python
async def discover_models() -> List[ModelInfo]:
    """Discover available Ollama models."""

async def validate_model(model_name: str) -> bool:
    """Validate model availability and readiness."""

async def pull_model(model_name: str) -> None:
    """Pull model from repository if not available."""
```

#### Health Management
```python
async def check_health() -> bool:
    """Check Ollama service health."""

async def warm_up_model(model_name: str) -> None:
    """Warm up model for optimal performance."""
```

### Configuration

```yaml
model_manager:
  base_url: "http://localhost:11434"
  connection_timeout: 30.0
  cache_duration: 3600
  max_retries: 3
  rate_limit: 10
```

### Error Handling

```python
class ModelError(Exception): pass
class ModelNotFoundError(ModelError): pass
class ModelConnectionError(ModelError): pass
class ModelValidationError(ModelError): pass
``` 