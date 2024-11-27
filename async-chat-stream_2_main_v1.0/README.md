# Async Chat Stream Processing System v1.0

An advanced Python-based chat and task processing application with Ollama model integration and dynamic interaction capabilities.

## Features

- Interactive chat sessions with Ollama models
- Task list processing and management
- Dynamic model discovery and categorization
- Flexible output organization by programming language
- Comprehensive logging system
- Robust error handling and recovery
- Asynchronous streaming responses
- Chat history management with JSON persistence
- Task list conversion and organization

## Requirements

- Python 3.8+
- Ollama server running locally (http://localhost:11434)
- Required Python packages:
  - ollama
  - asyncio
  - pathlib
  - logging
  - json
  - pandas
  - pydantic
  - dataclasses

## Project Structure

```
.
├── async-chat-stream_2_main_v1.0.py  # Main application script
├── logs/                             # Application logs
├── chats/                            # Saved chat histories
└── task_lists/                       # Task list storage
```

## Core Components

### Model Management

- Dynamic model discovery from Ollama API
- Model categorization by size:
  - Small (< 5GB)
  - Medium (5GB - 20GB)
  - Large (> 20GB)
  - Embedding models
- Automatic model list updates

### Chat System

- Interactive chat sessions
- Stream-based response processing
- Chat history saving/loading
- Multiple chat management
- Context preservation

### Task Processing

- Task list management
- CSV/JSON task storage
- Task categorization
- Progress tracking
- Output organization by language

### File Management

- Secure path handling
- Automatic folder creation
- Organized output structure
- Language-specific output sorting

## Usage

1. Start the Ollama server:
```bash
ollama serve
```

2. Run the application:
```bash
python async-chat-stream_2_main_v1.0.py
```

3. Main Menu Options:
   - Start Chat Session
   - Process Task List
   - View Available Models
   - View Current Tasks
   - Exit

## Error Handling

- Comprehensive error logging
- Connection failure recovery
- Input validation
- Path sanitization
- Resource cleanup

## Security Features

- Path traversal prevention
- Input sanitization
- Resource validation
- Secure file operations

## Data Management

- JSON-based chat storage
- CSV task list support
- Structured logging
- Metadata tracking

## Development Notes

- Type hints throughout codebase
- Comprehensive docstrings
- Error documentation
- Asynchronous design
- Resource management

## Future Enhancements

- Advanced task editing
- Enhanced search capabilities
- Multi-model chat sessions
- Advanced context management
- Performance optimizations

## Contributing

This is an educational project developed for the Hub City Python Users Group. Contributions and suggestions are welcome.

## License

Open source for educational and research purposes.
