# Chat Bot Technical Documentation

## Overview

This is an advanced AI chat bot implementation that interfaces with Ollama models, providing support for text chat, code generation, and image analysis capabilities. The bot features conversation history management, model selection, and persistent storage of chat sessions.

## Features

* Interactive model selection by category (Chat, Code, Vision, Embedding, Other)
* Persistent chat history storage in JSON format
* Support for vision models with image input
* Token-aware conversation history management
* Input sanitization and validation
* Comprehensive error handling and logging
* Session continuation from previous chats

## Dependencies

* langchain_ollama
* langchain_core
* tiktoken
* Ollama (local installation required)
* Python 3.7+
* pathlib
* logging
* subprocess
* json

## Core Components

### ChatBot Class

The main class handling chat functionality and state management.

#### Key Methods:

* `__init__()`: Initializes chat directory and settings
  * Creates chats directory
  * Sets max token history to 2000
  * Initializes history and model tracking
* `count_tokens(text: str) -> int`:
  * Uses tiktoken for accurate token counting
  * Falls back to word-based estimation if tiktoken fails
* `get_truncated_history() -> List[Dict]`:
  * Maintains conversation history within token limits
  * Returns most recent messages that fit within token budget
* `load_chat_history(chat_file: Path) -> None`:
  * Loads previous chat sessions from JSON
  * Handles file reading errors gracefully
* `save_chat_history() -> None`:
  * Persists chat sessions to JSON files
  * Includes error handling for file operations
* `get_available_chats() -> List[Path]`:
  * Returns list of existing chat history files
  * Searches for .json files in chats directory
* `create_new_chat(model_type: str) -> str`:
  * Creates new timestamped chat file
  * Initializes empty history
* `process_image_input() -> Optional[str]`:
  * Handles image URL or local path input
  * Validates image path existence
  * Returns None if user skips
* `select_model() -> Tuple[str, str]`:
  * Interactive model category and specific model selection
  * Returns category and model name

### Utility Functions

#### Model Management

* `get_available_ollama_models() -> List[str]`:
  * Retrieves installed Ollama models
  * Categorizes models by type (Chat, Code, Vision, etc.)
  * Uses subprocess to query Ollama CLI
* `display_available_models(models: dict) -> None`:
  * Formats and displays model options by category
  * Includes numbered listing for selection
* `validate_model(model_name: str) -> Optional[str]`:
  * Verifies model availability
  * Returns None for invalid models

#### Input Processing

* `sanitize_input(text: str) -> str`:
  * Removes special characters
  * Limits input length to 1000 characters
  * Preserves basic punctuation
* `setup_logging(log_dir: Path) -> logging.Logger`:
  * Configures file and console logging
  * Sets up formatted logging output
  * Creates log directory if needed
* `initialize_chat() -> str`:
  * Attempts to use preferred models first
  * Falls back to available models if needed
  * Raises error if no models available
* `handle_conversation() -> None`:
  * Main conversation loop
  * Manages model interactions
  * Handles user input/output
  * Logs conversation history

## File Structure

```
chat_bot_1/
├── chat_bot_1.py          # Main implementation
├── chat_bot_1.md          # Documentation
├── chats/                 # Stored chat histories
│   └── *.json            # Individual chat files
├── logs/                  # Log directory
│   └── chat_bot.log      # Application logs
└── conversations/         # Plain text conversation logs
    └── conversation_*.txt # Individual conversation logs
```

## Usage

### Starting a New Chat


1. Run the script: `python chat_bot_1.py`
2. Choose to start new chat or continue existing
3. Select model category and specific model
4. Begin conversation

### Commands

* `exit`: End the chat session
* `image`: Add image input (vision models only)

### Vision Model Support

For vision models, users can provide:

* Local image paths
* Image URLs
* Skip option to continue without image

## Error Handling

The system includes comprehensive error handling for:

* Model availability checks
* File I/O operations
* User input validation
* Token management
* Chat history operations
* Subprocess execution
* JSON parsing/writing

## Logging

Detailed logging includes:

* Info level: Normal operations (model selection, chat start/end)
* Warning level: Token counting failures, invalid inputs
* Error level: File operations, model errors
* Debug level: Message processing, history updates

## Future Improvements


 1. Streaming responses for real-time output
 2. Multi-modal conversation support
 3. Enhanced context management with embeddings
 4. Model-specific parameter tuning
 5. Chat export in multiple formats
 6. User preference persistence
 7. Custom model configuration support
 8. Batch image processing
 9. Conversation summarization
10. Integration with external knowledge bases

## Technical Notes

* Maximum token history: 2000 tokens (configurable)
* Default encoding: cl100k_base
* Conversation files include timestamps and process IDs
* Automatic directory creation for storage
* Sanitized input limited to 1000 characters
* JSON-based persistent storage
* Support for multiple model categories

## Security Considerations

* Input sanitization removes special characters
* File paths are validated before access
* JSON storage is protected against injection
* Model validation prevents unauthorized access
* Limited input length prevents buffer issues
* Secure file handling practices
* Protected logging implementation

## Performance Considerations

* History truncation maintains performance
* Efficient file handling with pathlib
* Modular design for scalability
* Minimal memory footprint
* Optimized token counting
* Smart history management
* Efficient model loading


---

Last Updated: 2024-11-22

Version: 1.0