# Ollama Chat Assistant

A sophisticated desktop chat application for interacting with Ollama models, featuring comprehensive model management, chat history, and an intuitive user interface.

## Features

### Model Management

* Dynamic model categorization (Chat, Code, Math, Vision, Embedding)
* Model filtering via dropdown menu
* Model details display (size, modification date)
* Copy model names to clipboard
* Automatic model list updates

### Chat Interface

* Real-time streaming responses
* Subject-based chat organization
* Chat history management
* Copy/paste support
* Clear chat functionality

### History Management

* JSON and text-based chat storage
* Load previous conversations
* Delete chat histories
* Automatic chat saving

## Technical Details

### Core Classes

#### ChatBotApp

Main application class managing the UI and core functionality.

**Key Methods:**

* `__init__`: Initializes application components and UI
* `setup_ui`: Creates main UI layout
* `setup_left_panel`: Creates model selection interface
* `setup_middle_panel`: Creates chat interface
* `setup_right_panel`: Creates chat history interface
* `load_models`: Loads and categorizes Ollama models
* `handle_chat`: Manages async chat interactions
* `save_current_chat`: Saves chat history
* `display_filtered_models`: Shows filtered model list

#### ChatHistory

Manages chat history storage and retrieval.

**Key Methods:**

* `save_chat`: Saves chat in JSON and text formats
* `load_chat`: Loads previous chat sessions
* `_ensure_directories`: Creates necessary storage directories

### File Structure

```
chat_bot_2_recursive/
├── chat_bot_2_recursive.py  # Main application
├── models/                  # Model information
│   ├── ollama_models.json
│   ├── ollama_vision_models.json
│   ├── ollama_tool_use_models.json
│   └── ollama_embeddings_models.json
├── logs/                   # Application logs
│   └── chat_bot.log
├── json_outputs/          # Chat history (JSON)
└── convos/               # Chat history (Text)
```

### Dependencies

* Python 3.10+
* tkinter
* ollama
* nltk
* pathlib
* logging

## Implementation Details

### Error Handling

* Comprehensive exception handling
* Detailed logging
* User-friendly error messages
* Graceful failure recovery

### Data Management

* Strong typing throughout
* Input validation
* Path management
* Configuration handling

### UI Components

* Model selection dropdown
* Chat display area
* Input field with history
* Chat history viewer
* Control buttons

## Current Progress

### Completed

- [x] Basic UI implementation
- [x] Model categorization
- [x] Chat functionality
- [x] History management
- [x] Logging system
- [x] Error handling
- [x] Copy/paste support
- [x] Model filtering

### To Do

- [ ] Add model search functionality
- [ ] Implement model parameter customization
- [ ] Add conversation context management
- [ ] Implement chat export functionality
- [ ] Add model performance metrics
- [ ] Implement user preferences
- [ ] Add keyboard shortcuts
- [ ] Implement dark/light theme
- [ ] Add conversation branching
- [ ] Implement model comparison tools
- [ ] Add batch processing capabilities
- [ ] Implement chat summarization
- [ ] Add conversation tagging
- [ ] Implement model update notifications
- [ ] Add chat encryption options

## Usage


1. Start Ollama server:

```bash
ollama serve
```


2. Run the application:

```bash
python chat_bot_2_recursive.py
```


3. Select a model from the dropdown menu
4. Enter a subject for your chat
5. Start chatting!

## Contributing


1. Fork the repository
2. Create a feature branch
3. Follow code standards:
   * Use type hints
   * Add docstrings
   * Include error handling
   * Write tests
4. Submit a pull request

## License

\[Specify License\]

## Troubleshooting

Common issues and solutions:


1. **Model Loading Issues**
   * Check Ollama server status
   * Verify model installation
   * Check network connection
2. **UI Issues**
   * Verify tkinter installation
   * Check Python version
   * Verify display settings
3. **History Issues**
   * Check file permissions
   * Verify storage paths
   * Check disk space


