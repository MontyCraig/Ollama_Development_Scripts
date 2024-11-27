# Async Chat Stream v1.2.2

A robust asynchronous chat application that interfaces with Ollama models to process tasks and generate responses across multiple programming languages.

## Technical Overview

### Core Features

- Asynchronous task processing using `asyncio`
- Streaming responses from Ollama AI models
- Multi-language code output handling (JavaScript, PHP, Python, HTML/CSS, SQL)
- Structured output organization with automatic folder creation
- CSV-based task management system
- Real-time text-to-speech capability (optional)
- Comprehensive logging and metadata tracking

### Architecture

#### Main Components

1. **Task Management**
   - CSV parsing and reformatting (`reformat_csv`)
   - Hierarchical task organization with groups
   - Interactive task selection interface

2. **Output Management**
   - Dynamic folder creation for different languages
   - Automatic file versioning
   - Structured output paths for various file types:
     - Text output
     - JSON output
     - Language-specific code outputs

3. **AI Integration**
   - Asynchronous Ollama client implementation
   - Streaming response handling
   - Context management with pre/post prompts
   - Token counting and metadata tracking

4. **Error Handling**
   - File operation error management
   - Input validation
   - Graceful exit handling

### Technical Specifications

- **Python Version**: 3.8+
- **Key Dependencies**:
  - ollama
  - asyncio
  - pandas
  - json
  - pathlib

## Code Standards Compliance

The current implementation follows most of our established standards, but requires the following improvements:

1. **Type Hints**
   - Add type hints to all function parameters
   - Implement return type annotations
   - Define custom types for complex structures

2. **Documentation**
   - Add detailed docstrings to all functions
   - Include usage examples
   - Document exception cases

3. **Error Handling**
   - Implement custom exception classes
   - Add more specific error messages
   - Enhance error recovery mechanisms

## Suggested Upgrades

### Immediate Improvements

1. **Code Structure**
   - Split into smaller modules (current file exceeds 300 lines)
   - Create separate modules for:
     - Task management
     - Output handling
     - AI interaction
     - CLI interface

2. **Type Safety**
   ```python
   from typing import Dict, List, Optional, Union, AsyncIterator
   
   async def process_task(
       client: ollama.AsyncClient,
       task: str,
       output_folders: Dict[str, str],
       languages: List[str],
       speaker: Optional[str],
       model: str
   ) -> Dict[str, Union[str, int, float]]
   ```

3. **Configuration Management**
   - Move hardcoded values to config files
   - Implement environment variable support
   - Add configuration validation

### Future Enhancements

1. **Performance Optimization**
   - Implement response caching
   - Add batch processing capabilities
   - Optimize file I/O operations

2. **User Interface**
   - Add web interface option
   - Implement progress bars
   - Enhanced CLI with rich text support

3. **Testing**
   - Add unit tests
   - Implement integration tests
   - Add test coverage reporting

4. **Security**
   - Add input sanitization
   - Implement rate limiting
   - Add authentication for sensitive operations

## To-Do List

### Priority 1 (Immediate)
- [ ] Add type hints to all functions
- [ ] Create comprehensive docstrings
- [ ] Split code into modules
- [ ] Implement custom exceptions
- [ ] Add input validation

### Priority 2 (Short-term)
- [ ] Create configuration system
- [ ] Add logging system
- [ ] Implement basic tests
- [ ] Add progress indicators
- [ ] Create setup.py

### Priority 3 (Long-term)
- [ ] Develop web interface
- [ ] Add authentication system
- [ ] Implement caching
- [ ] Create API documentation
- [ ] Add performance monitoring

## Installation

```bash
# Clone the repository
git clone [repository-url]

# Install dependencies
pip install -r requirements.txt

# Run the application
python async_chat_stream_2_main_v1.2.2.py
```

## Usage

```bash
# Basic usage
python async_chat_stream_2_main_v1.2.2.py

# With text-to-speech enabled
python async_chat_stream_2_main_v1.2.2.py --speak

# With specific model
python async_chat_stream_2_main_v1.2.2.py --model qwen2:7b
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## License

[Specify License]
