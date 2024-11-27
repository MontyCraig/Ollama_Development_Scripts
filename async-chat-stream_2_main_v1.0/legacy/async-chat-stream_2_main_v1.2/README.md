# Async Chat Stream v1.2

An enhanced version of the asynchronous chat application with improved configuration management, advanced error handling, and robust task processing capabilities.

## Technical Overview

### Core Features

* YAML-based configuration management
* Advanced error handling with detailed logging
* Asynchronous HTTP client implementation (aiohttp)
* Automatic model pulling and validation
* Progress tracking with tqdm
* Task completion persistence
* Resume capability for interrupted sessions
* Multi-format output management

### Architecture

#### Key Components


1. **Configuration Management**
   * YAML-based configuration file
   * Default fallback values
   * Environment-specific settings
   * Runtime configuration validation
2. **Task Processing**
   * Paginated task display
   * Task completion tracking
   * Progress persistence
   * Batch processing support
   * Resume functionality
3. **Error Recovery**
   * Automatic model pulling
   * Connection retry logic
   * Graceful error handling
   * Detailed error logging
4. **Output Management**
   * Structured folder hierarchy
   * Multiple output formats
   * Version control
   * Metadata tracking

### Technical Specifications

* **Python Version**: 3.8+
* **Key Dependencies**:
  * ollama
  * aiohttp
  * pyyaml
  * tqdm
  * pandas
  * asyncio

## Configuration

```yaml
# config.yaml
pre_prompt_path: "/path/to/pre_prompt.txt"
post_prompt_path: "/path/to/post_prompt.txt"
default_model: "llama2"
input_csv_path: "/path/to/input.csv"
output_csv_path: "output.csv"
output_base_path: "output"
```

## Installation

```bash
# Clone the repository
git clone [repository-url]

# Install dependencies
pip install -r requirements.txt

# Create configuration
cp config.yaml.example config.yaml
```

## Usage

```bash
# Basic usage
python async-chat-stream_2_main_v1.2.py

# With specific model
python async-chat-stream_2_main_v1.2.py --model qwen2:7b

# Enable text-to-speech
python async-chat-stream_2_main_v1.2.py --speak
```

## Features

### Task Management

* Paginated task display (10 tasks per page)
* Task status tracking (Completed/Pending)
* Batch processing capabilities
* Session persistence
* Resume from last task

### Error Handling

* Automatic model availability check
* Model pulling if not available
* Connection error recovery
* Detailed error logging
* User-friendly error messages

### Output Organization

* Hierarchical folder structure:

  ```
  output/
  ├── chatbot_outputs/
  │   └── model_tests/
  │       └── [model_name]/
  │           └── [output_type]/
  │               ├── json_output/
  │               ├── txt_output/
  │               └── [language]_output/
  ```

## Code Standards

The implementation follows strict coding standards as defined in CODE_STANDARDS.md:


1. **Error Handling**
   * Custom exceptions for specific errors
   * Comprehensive error logging
   * Graceful degradation
2. **Configuration**
   * No hardcoded values
   * Environment-specific settings
   * Configuration validation
3. **Logging**
   * Hierarchical logging levels
   * Contextual information
   * Error traceability

## To-Do List

### Priority 1 (Immediate)

- [ ] Add input sanitization
- [ ] Implement rate limiting
- [ ] Add model response caching
- [ ] Enhance error recovery

### Priority 2 (Short-term)

- [ ] Add API endpoints
- [ ] Implement user authentication
- [ ] Add concurrent processing
- [ ] Enhance progress tracking

### Priority 3 (Long-term)

- [ ] Create web interface
- [ ] Add monitoring dashboard
- [ ] Implement load balancing
- [ ] Add distributed processing

## Contributing


1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit pull request
5. Follow code review process

## License

\[Specify License\]

## Notes

* Ensure config.yaml is properly configured
* Check model availability before running
* Configure logging appropriately
* Review security settings for production use
* Backup task completion data regularly

## Changelog from v1.0

* Added YAML configuration support
* Implemented task completion persistence
* Added session resume capability
* Enhanced error handling and recovery
* Improved task display with pagination
* Added automatic model pulling
* Enhanced logging system
* Improved folder structure
* Added progress tracking


