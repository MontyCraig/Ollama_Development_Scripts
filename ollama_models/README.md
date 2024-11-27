# Ollama Models Configuration

## Overview

This directory contains configuration and specification files for Ollama models, automatically updated when scripts interact with the local Ollama installation.

## Configuration Files

### ollama_models_details.json
Detailed specifications for all available models:
* Model sizes and requirements
* Context window sizes
* Specific capabilities
* Performance characteristics

### ollama_embeddings_models.json
Models specifically capable of generating embeddings:
* Embedding dimensions
* Supported formats
* Usage requirements
* Performance metrics

### ollama_tool_use_models.json
Models with tool use capabilities:
* Available tools
* Integration methods
* Usage examples
* Compatibility information

### ollama_vision_models.json
Models supporting vision tasks:
* Image processing capabilities
* Input format requirements
* Resolution specifications
* Performance guidelines

### ollama_available_models.csv
Current snapshot of locally available models:
* Model names
* Installation status
* Last update timestamp
* Local configurations

## Auto-Update Functionality

The configuration files are automatically updated when:
* Running any script that queries Ollama
* Checking local model availability
* Installing new models
* Updating existing models

## Usage

### In Legacy Scripts
```python
# Example from async_chat_stream_2_main_v1.2.2.py
with open('ollama_models/ollama_models_details.json', 'r') as f:
    model_details = json.load(f)
```

### In New Modular Structure
```python
from modules.models.model_manager import ModelManager

model_manager = ModelManager()
model_details = model_manager.get_model_details()
```

### Tool Function Support

Models can now use Python functions as tools. Example usage:

```python
from modules.models.model_manager import ModelManager

def add_numbers(a: int, b: int) -> int:
    """
    Add two numbers together.
    
    Args:
        a: The first integer number
        b: The second integer number
        
    Returns:
        The sum of the two numbers
    """
    return a + b

# Register the function as a tool
model_manager = ModelManager()
model_manager.register_tool(add_numbers)
```

## Tool Development Roadmap

### Code Generation & Analysis Tools
- [ ] Code completion tool
  * Context-aware code suggestions
  * Multiple language support
  * Style guide compliance
- [ ] Code review tool
  * Static analysis
  * Best practices checking
  * Security vulnerability scanning
- [ ] Documentation generator
  * Docstring generation
  * README creation
  * API documentation
- [ ] Test generator
  * Unit test creation
  * Test case generation
  * Coverage analysis

### Web Interaction Tools
- [ ] Web scraping tools
  * HTML content extraction
  * Data structure recognition
  * Rate limiting and caching
- [ ] API interaction tools
  * REST API client
  * GraphQL support
  * Authentication handling
- [ ] Search tools
  * Web search integration
  * Documentation search
  * Code repository search

### Development Tools
- [ ] Git operations
  * Commit message generation
  * Code diff analysis
  * Branch management
- [ ] Database tools
  * Query generation
  * Schema analysis
  * Data validation
- [ ] Environment management
  * Dependency tracking
  * Virtual environment handling
  * Package management

### Communication Tools
- [ ] Email tools
  * Template generation
  * Email parsing
  * Attachment handling
- [ ] Documentation tools
  * Markdown formatting
  * Technical writing assistance
  * Translation support
- [ ] Collaboration tools
  * Issue tracking integration
  * PR description generation
  * Code review comments

### System Tools
- [ ] File operations
  * File format conversion
  * Directory organization
  * Backup management
- [ ] Process management
  * Resource monitoring
  * Log analysis
  * Performance tracking
- [ ] Configuration tools
  * Config file generation
  * Environment setup
  * Validation checks

### AI Integration Tools
- [ ] Model management
  * Model selection
  * Parameter optimization
  * Performance monitoring
- [ ] Training tools
  * Dataset preparation
  * Fine-tuning automation
  * Evaluation metrics
- [ ] Pipeline tools
  * Workflow automation
  * Task chaining
  * Error handling

### Security Tools
- [ ] Code security
  * Vulnerability scanning
  * Dependency checking
  * License compliance
- [ ] Authentication tools
  * Token management
  * Permission validation
  * Access control
- [ ] Data protection
  * Encryption/decryption
  * Data sanitization
  * PII detection

## Implementation Priority

### Phase 1 (Core Tools)
1. Code completion and analysis
2. Web scraping and API interaction
3. Git operations
4. File management
5. Basic security tools

### Phase 2 (Enhancement Tools)
1. Documentation generation
2. Test creation
3. Database operations
4. Email integration
5. Configuration management

### Phase 3 (Advanced Tools)
1. AI model management
2. Pipeline automation
3. Advanced security
4. Performance optimization
5. Collaboration features

## Tool Development Guidelines

1. **Documentation Requirements**
   * Comprehensive docstrings
   * Type hints
   * Usage examples
   * Error handling documentation

2. **Security Considerations**
   * Input validation
   * Rate limiting
   * Credential management
   * Audit logging

3. **Performance Standards**
   * Asynchronous where applicable
   * Resource usage optimization
   * Caching strategies
   * Error recovery

4. **Integration Requirements**
   * Modular design
   * Standard interfaces
   * Event logging
   * Metric collection

## To-Do List

### Priority 1 (Immediate)
- [ ] Add model version tracking
- [ ] Implement automatic updates on model changes
- [ ] Add model compatibility checks

### Priority 2 (Short-term)
- [ ] Add model performance metrics
- [ ] Implement model comparison tools
- [ ] Create model selection recommendations

### Priority 3 (Long-term)
- [ ] Add support for custom model configurations
- [ ] Implement model usage analytics
- [ ] Create model performance benchmarks
- [ ] Add automatic model capability detection 