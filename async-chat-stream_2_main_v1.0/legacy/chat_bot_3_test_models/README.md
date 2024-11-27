# Chat Bot Model Testing Framework

A comprehensive testing framework for evaluating multiple Ollama models with standardized prompts and detailed performance metrics.

## Technical Overview

### Core Features

* Automated testing of multiple Ollama models
* Comprehensive performance metrics collection
* Path-safe file operations using pathlib
* Robust error handling and logging
* Input sanitization and validation
* Structured output organization
* Progress tracking and reporting

### Architecture

#### Key Components


1. **Model Management**
   * Automatic model discovery
   * Connection validation
   * Model availability checking
   * Response timeout handling
2. **Testing Framework**
   * Multi-line prompt support
   * Batch model processing
   * Progress tracking
   * Rate limiting
   * Performance metrics
3. **Output Management**
   * Structured output directories
   * JSON and TXT output formats
   * Timestamped results
   * Detailed metrics logging
4. **Error Handling**
   * Custom exception types
   * Comprehensive error logging
   * Graceful degradation
   * Connection recovery

### Technical Specifications

* **Python Version**: 3.8+
* **Dependencies**:
  * ollama
  * nltk
  * pathlib
  * typing_extensions
  * logging

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

## Usage

```bash
# Run the testing framework
python chat_bot_3_test_models.py
```

## Features

### Model Testing

* Automated model discovery
* Response time measurement
* Token counting
* Multi-line prompt support
* Progress tracking
* Rate limiting

### Output Organization

```
model_tests/
├── [prompt_subject]_[timestamp]/
│   ├── json/
│   │   └── [model_name]_[prompt_subject].json
│   └── txt/
│       └── [model_name]_[prompt_subject].txt
```

## Integration To-Do List

### Priority 1 (Immediate)

- [ ] Create ModelTester class to encapsulate testing logic
- [ ] Implement custom exceptions for model testing
- [ ] Add configuration file support
- [ ] Create separate modules for:
  * Model management
  * Testing framework
  * Output handling
  * Metrics collection

### Priority 2 (Short-term)

- [ ] Integrate with async-chat-stream codebase
- [ ] Add model comparison features
- [ ] Implement result visualization
- [ ] Add test result database
- [ ] Create API endpoints for testing

### Priority 3 (Long-term)

- [ ] Add distributed testing capability
- [ ] Implement model performance analytics
- [ ] Create web interface for testing
- [ ] Add automated test scheduling
- [ ] Implement test result sharing

## Code Refactoring Plan


1. **Create Module Structure**

```
chat_bot/
├── core/
│   ├── __init__.py
│   ├── model_manager.py
│   ├── test_framework.py
│   └── output_handler.py
├── utils/
│   ├── __init__.py
│   ├── logging.py
│   └── metrics.py
└── tests/
    ├── __init__.py
    └── test_models.py
```


2. **Class Structure**

```python
class ModelManager:
    """Handles model discovery and management"""
    
class TestFramework:
    """Manages test execution and metrics"""
    
class OutputHandler:
    """Handles result storage and organization"""
    
class MetricsCollector:
    """Collects and analyzes test metrics"""
```


3. **Integration Steps**

- [ ] Extract model testing logic to ModelTester class
- [ ] Create configuration management system
- [ ] Implement metrics collection framework
- [ ] Add result storage and retrieval system
- [ ] Create API for external integration
- [ ] Add documentation and examples

## Contributing


1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit pull request
5. Follow code review process

## License

\[Specify License\]

## Notes

* Ensure Ollama is running before testing
* Configure logging appropriately
* Review rate limiting settings
* Backup test results regularly
* Monitor system resources during testing


