# Async Chat Stream v1

A Python-based asynchronous chat application that provides advanced task processing and AI model interaction capabilities using Ollama.

## Technical Overview

### Core Features

* Robust error handling and logging system
* Type-safe implementation with comprehensive type hints
* Custom exception hierarchy for specific error cases
* Structured file and folder management
* Asynchronous chat processing with streaming support
* Multi-language code output handling
* CSV-based task management
* Text-to-speech integration (optional)

### Architecture

#### Key Components


1. **Exception Handling**
   * Custom exception hierarchy:
     * `ChatStreamError` (base)
     * `TaskProcessingError`
     * `OutputError`
   * Comprehensive error logging and recovery
2. **File Management**
   * Secure path handling and sanitization
   * Automatic folder creation and organization
   * Version control for output files
   * Support for multiple output formats (JSON, TXT, language-specific)
3. **Task Processing**
   * CSV parsing and reformatting
   * Task grouping and organization
   * Interactive task selection
   * Batch processing capabilities
4. **Model Integration**
   * Asynchronous Ollama client implementation
   * Model categorization (size and type)
   * Embedding model support
   * Streaming response handling

### Technical Specifications

* **Python Version**: 3.8+
* **Dependencies**:
  * ollama
  * asyncio
  * pandas
  * pathlib
  * logging

## Code Standards

The implementation adheres to strict coding standards:


1. **Type Safety**
   * Comprehensive type hints
   * Custom type definitions
   * Runtime type checking
2. **Documentation**
   * Detailed function docstrings
   * Inline code comments
   * Exception documentation
3. **Error Handling**
   * Custom exception classes
   * Contextual error messages
   * Proper error propagation
4. **Logging**
   * Hierarchical logging system
   * Detailed error tracking
   * Debug information capture

## Installation

```bash
# Clone the repository
git clone [repository-url]

# Install dependencies
pip install -r requirements.txt

# Set up logging directory
mkdir -p logs
```

## Usage

```bash
# Basic usage
python async-chat-stream_2_main_v1.py

# Enable text-to-speech
python async-chat-stream_2_main_v1.py --speak

# Specify model
python async-chat-stream_2_main_v1.py --model qwen2:7b
```

## Configuration

The application supports several configuration options:


1. **Logging**
   * Log level: INFO (default)
   * Log format: Timestamp, module, level, message
   * Output: File and console
2. **Model Selection**
   * Size categories: SMALL, MEDIUM, LARGE
   * Embedding models support
   * Custom model configuration
3. **Output Management**
   * Configurable output directories
   * Multiple format support
   * Version control

## To-Do List

### Priority 1 (Immediate)

- [ ] Add configuration file support
- [ ] Implement rate limiting
- [ ] Enhance error recovery mechanisms
- [ ] Add input validation layer

### Priority 2 (Short-term)

- [ ] Create web interface
- [ ] Add user authentication
- [ ] Implement session management
- [ ] Add response caching

### Priority 3 (Long-term)

- [ ] Develop plugin system
- [ ] Add multi-model support
- [ ] Implement advanced logging
- [ ] Create monitoring dashboard

## Contributing


1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit pull request
5. Follow code review process

## License

\[Specify License\]

## Notes

* Ensure proper model availability before running
* Check system requirements for text-to-speech
* Configure logging paths appropriately
* Review security settings for production use


