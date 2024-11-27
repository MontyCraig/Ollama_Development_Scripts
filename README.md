# Async Chat Stream Processing System

## Overview
An advanced asynchronous chat interface that leverages Ollama's AI models for intelligent task processing and code generation. The system features model size categorization, embedding support, and comprehensive task management.

## Key Features

### Model Management
- Dynamic model categorization (Small < 5GB, Medium 5-20GB, Large > 20GB)
- Embedding model support
- Real-time model list updates
- Size-based model filtering

### Chat Interface
- Interactive terminal-based chat
- Asynchronous response streaming
- Conversation saving capabilities
- Task list generation from chats

### Task Processing
- CSV-based task management
- Hierarchical task organization
- Task status tracking
- Automated task reformatting

### Output Management
- Multi-format output generation
  - Text files
  - JSON metadata
  - Language-specific code files
- UUID-based file organization
- Version control for outputs

### System Features
- Comprehensive logging
- Path sanitization
- Error handling
- Metadata tracking
- Async I/O operations

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python async-chat-stream_2_main_v1.0.py
```

## Menu Options
1. Start Chat Session
2. Process Task List
3. View Available Models
4. View Current Tasks
5. Exit

## Usage Examples

### Chat Mode
```bash
1. Select "Start Chat Session"
2. Choose model size preference
3. Select specific model
4. Start chatting
5. Type 'save' to save conversation
6. Type 'exit' to end chat
```

### Task Processing
```bash
1. Select "Process Task List"
2. Choose model
3. View task execution
4. Check output in respective folders
```

## Project Structure
```
async-chat-stream_2_main_v1.0/
├── async-chat-stream_2_main_v1.0.py
├── requirements.txt
├── README.md
├── SETUP.md
├── output/
│   └── model_tests/
├── task_lists/
└── async_chat_stream.log
```

## Documentation
- [Setup Guide](SETUP.md)
- [Technical Report](TECHNICAL_REPORT.md)
- [Code Analysis](CODE_ANALYSIS.md)

## Requirements
- Python 3.8+
- Ollama API
- See requirements.txt for full list

## License
[Specify License]

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```

</rewritten_file>