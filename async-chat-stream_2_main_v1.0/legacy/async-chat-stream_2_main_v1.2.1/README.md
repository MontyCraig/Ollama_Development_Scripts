# Async Chat Stream v1.2.1

An enhanced version of the asynchronous chat application with improved path handling, type safety, and more robust error management using pathlib.

## Technical Overview

### Core Features

* Path-safe file operations using pathlib.Path
* Enhanced type hints and validation
* Relative path handling from script directory
* Improved error handling and logging
* Asynchronous HTTP client implementation
* Task completion persistence
* Resume capability for interrupted sessions
* Multi-format output management

### Architecture

#### Key Components


1. **Path Management**
   * pathlib.Path for all file operations
   * Script-relative path resolution
   * Automatic directory creation
   * Path existence validation
   * Safe path joining operations
2. **Type System**
   * Comprehensive type hints
   * Union types for flexible inputs
   * Optional type handling
   * Return type annotations
   * Type validation at runtime
3. **Error Recovery**
   * Enhanced error context
   * Path-specific error handling
   * Graceful degradation
   * Detailed logging with paths
4. **Output Management**
   * Type-safe file operations
   * Atomic writes where possible
   * Safe file versioning
   * Metadata tracking

### Technical Specifications

* **Python Version**: 3.8+
* **Key Dependencies**:
  * ollama
  * aiohttp
  * pyyaml
  * tqdm
  * pandas
  * pathlib
  * typing_extensions

## Configuration

```yaml
# config.yaml
pre_prompt_path: "config/pre_prompt.txt"
post_prompt_path: "config/post_prompt.txt"
default_model: "llama2"
input_csv_path: "input/input.csv"
output_csv_path: "output/output.csv"
output_base_path: "output"
```

## Installation

```bash
# Clone the repository
git clone [repository-url]

# Install dependencies
pip install -r requirements.txt

# Create required directories
mkdir -p config input output logs
```

## Usage

```bash
# Basic usage
python async-chat-stream_2_main_v1.2.1.py

# With specific model
python async-chat-stream_2_main_v1.2.1.py --model qwen2:7b

# Enable text-to-speech
python async-chat-stream_2_main_v1.2.1.py --speak
```

## Features

### Path Management

* Script-relative path resolution
* Automatic directory creation
* Path validation and sanitization
* Safe path joining and manipulation
* Cross-platform path handling

### Type Safety

* Function parameter type hints
* Return type annotations
* Union types for flexibility
* Optional type handling
* Runtime type checking

### Error Handling

* Path-specific error types
* Enhanced error context
* Graceful fallbacks
* Detailed error logging
* User-friendly messages

## Code Standards

The implementation strictly follows the standards in CODE_STANDARDS.md:


1. **Path Operations**
   * Use pathlib.Path exclusively
   * Validate paths before operations
   * Handle permissions appropriately
   * Ensure directory existence
2. **Type Safety**
   * Complete type annotations
   * Union types where needed
   * Optional for nullable values
   * Return type specifications
3. **Error Management**
   * Path-specific exceptions
   * Detailed error context
   * Proper error propagation
   * Comprehensive logging

## To-Do List

### Priority 1 (Immediate)

- [ ] Add path sanitization tests
- [ ] Implement path permission checks
- [ ] Add atomic file operations
- [ ] Enhance path validation

### Priority 2 (Short-term)

- [ ] Add path caching
- [ ] Implement path monitoring
- [ ] Add path cleanup utilities
- [ ] Enhance cross-platform support

### Priority 3 (Long-term)

- [ ] Create path abstraction layer
- [ ] Add distributed file support
- [ ] Implement path optimization
- [ ] Add path security features

## Contributing


1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit pull request
5. Follow code review process

## License

\[Specify License\]

## Notes

* Ensure proper path permissions
* Check directory structure
* Configure logging paths
* Review path security settings
* Backup completion data

## Changelog from v1.2

* Converted to pathlib.Path for all file operations
* Added comprehensive type hints
* Enhanced path validation and security
* Improved error handling for paths
* Added script-relative path resolution
* Enhanced logging with path context
* Improved cross-platform compatibility
* Added automatic directory creation
* Enhanced path-based error recovery


