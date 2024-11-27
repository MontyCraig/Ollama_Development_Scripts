# Chat Bot 2 Recursive

A production-grade chatbot application that interfaces with Ollama models, featuring comprehensive error handling, logging, and conversation management.

## Latest Updates

* Improved response streaming stability
* Enhanced file path handling
* Better error recovery
* UTF-8 encoding support
* Sanitized filename handling

## Features

### Core Functionality

* Interactive chat interface with multiple model support
* Conversation history management
* Context preservation
* Real-time response streaming
* Category-based model selection
* Chat subject tracking

### Technical Features

* Strong type hints throughout
* Comprehensive error handling
* Detailed logging system
* Input validation and sanitization
* File operation safety
* Path management

## Installation


1. Set up virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```


2. Install dependencies:

```bash
pip install -r requirements.txt
```


3. Install Ollama:

```bash
# On macOS/Linux
curl https://ollama.ai/install.sh | sh

# Or visit https://ollama.ai/download for other platforms
```

## Project Structure

```
chat_bot_2_recursive/
├── chat_bot_2_recursive.py  # Main application
├── chat_bot.log            # Application logs
├── ollama_models.json      # Cached model list
├── json_outputs/          # JSON conversation archives
└── convos/               # Text conversation archives
```

## Usage


1. Start Ollama server:

```bash
ollama serve
```


2. Pull desired models:

```bash
ollama pull model_name
```


3. Run the chatbot:

```bash
python chat_bot_2_recursive.py
```

## Model Configuration

Default model parameters:

```python
options = {
    "timeout": 60,        # Response timeout in seconds
    "temperature": 0.7,   # Response creativity (0.0-1.0)
    "top_k": 40,         # Top K sampling
    "top_p": 0.9         # Nucleus sampling threshold
}
```

## Error Handling

The application handles various scenarios:

* Network connectivity issues
* Model availability problems
* Invalid inputs
* File operation failures
* Memory constraints
* Timeout scenarios
* Malformed responses

## File Storage

### JSON Format

```json
{
    "timestamp": "2024-11-24 14:05:51",
    "user_message": "Hello!",
    "bot_response": "Hi! How can I help you?",
    "model": "model_name",
    "subject": "chat_subject"
}
```

### Text Format

```text
Model: model_name
Date: 2024-11-24 14:05:51
Subject: chat_subject

2024-11-24 14:05:51 - User: Hello!
2024-11-24 14:05:51 - Bot: Hi! How can I help you?
```

## Commands

During chat:

* `exit` - End conversation
* `show details` - Display model information
* `clear context` - Reset conversation context

## Logging

Detailed logging with format:

```python
'%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
```

Log levels:

* DEBUG: Detailed debugging information
* INFO: General operational events
* WARNING: Unexpected but handled events
* ERROR: Serious issues that need attention
* CRITICAL: System-level failures

## Development

### Type Safety

```python
def stream_response(
    model_name: str,
    messages: List[Dict[str, str]],
    timeout: int = 60
) -> str:
    """Stream and aggregate response from Ollama model."""
```

### Error Recovery

```python
try:
    response = stream_response(model_name, messages)
except TimeoutError:
    logger.error("Response timed out")
    return "I apologize, but the response timed out."
except Exception as e:
    logger.error(f"Error: {str(e)}")
    raise
```

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


1. **Model Timeout**
   * Increase timeout value
   * Check network connection


