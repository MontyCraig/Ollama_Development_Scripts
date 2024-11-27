# Async Chat Stream Processing System v1.0.1

An enhanced Python-based chat and task processing system featuring robust Ollama model integration, improved async handling, and advanced error recovery.

## Key Improvements in v1.0.1

- Enhanced async model handling with proper AsyncClient implementation
- Improved model discovery and categorization system
- Robust error handling for Ollama server connections
- Better path handling and security measures
- Streamlined chat session management
- Enhanced logging system with detailed error tracking

## Core Features

### Model Management
- Dynamic model discovery via Ollama API
- Intelligent model categorization:
  - Embedding models (text-embedding, bge, e5, etc.)
  - Size-based categories (Small < 5GB, Medium 5-20GB, Large > 20GB)
- Automatic model list updates with metadata tracking
- Robust connection handling and fallbacks

### Chat System
- Asynchronous streaming responses
- Improved chat history management
- Session persistence with JSON storage
- Enhanced conversation context handling
- Multiple chat management capabilities

### Task Processing
- Structured task list management
- CSV/JSON format support
- Intelligent task categorization
- Progress tracking
- Multi-language output organization

## Technical Requirements

### System Requirements
- Python 3.8+
- Ollama server (running on localhost:11434)
- macOS/Linux environment
- Sufficient storage for model files

### Python Dependencies
```
ollama
asyncio
pathlib
logging
json
pandas
pydantic
dataclasses
```

## Project Structure

```
.
├── async-chat-stream_2_main_v1.01.py  # Main application script
├── logs/                              # Application logs
│   └── async_chat_stream.log          # Detailed logging
├── chats/                             # Chat history storage
│   └── *.json                         # Chat session files
├── task_lists/                        # Task management
│   └── *.csv                          # Task list files
└── ollama_models_list.txt             # Current model catalog
```

## Security Features

- Path traversal prevention
- Input sanitization
- Resource validation
- Secure file operations
- Error isolation
- Permission management

## Usage Guide

1. Start the Ollama server:
```bash
ollama serve
```

2. Launch the application:
```bash
python async-chat-stream_2_main_v1.01.py
```

3. Main Menu Options:
   - Start Chat Session: Begin interactive chat with selected model
   - Process Task List: Handle task processing workflows
   - View Available Models: Browse and select models
   - View Current Tasks: Manage task lists
   - Exit: Safely terminate the application

4. Model Selection:
   - Choose model category (Small/Medium/Large/Embedding)
   - Select specific model from available options
   - View model details (size, category, capabilities)

5. Chat Operations:
   - Interactive chat with streaming responses
   - Save/load chat histories
   - Convert chats to task lists
   - Manage multiple chat sessions

## Error Handling

- Comprehensive error logging
- Graceful connection failure recovery
- Input validation at all levels
- Resource cleanup on errors
- Detailed error messages and suggestions

## Data Management

- Structured logging system
- JSON-based chat storage
- CSV task list format
- Metadata tracking
- Session persistence

## Development Notes

- Type hints throughout codebase
- Comprehensive docstrings
- Async/await pattern implementation
- Resource management
- Error documentation

## Future Enhancements

- Multi-model chat sessions
- Advanced context management
- Enhanced search capabilities
- Performance optimizations
- Advanced task editing features

## Contributing

This is an educational project developed for the Hub City Python Users Group. Contributions and suggestions are welcome through:
- Feature requests
- Bug reports
- Documentation improvements
- Code contributions

## License

Open source for educational and research purposes.

## Acknowledgments

- Hub City Python Users Group
- Ollama development team
- Python async community
