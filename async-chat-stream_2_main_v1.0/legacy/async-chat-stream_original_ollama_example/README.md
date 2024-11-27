# Async Chat Stream (Ollama Example)

A simple demonstration of creating an interactive chat interface using Ollama's asynchronous client with streaming responses and optional text-to-speech capabilities.

## Overview

This example demonstrates:

* Asynchronous chat interactions with Ollama models
* Streaming response handling
* Text-to-speech integration
* Basic conversation history management
* Cross-platform TTS support (macOS and Linux)

## Features

### Core Functionality

* Real-time streaming of model responses
* Conversation history tracking
* Sentence-level TTS processing
* Clean exit handling with Ctrl+C

### Text-to-Speech Support

* macOS: Uses built-in `say` command
* Linux: Supports both `espeak` and `espeak-ng`
* Automatic TTS system detection
* Sentence-by-sentence speech output

## Technical Details

### Dependencies

* Python 3.8+
* ollama
* asyncio
* argparse

### Model

* Default model: 'mistral'
* Streaming enabled by default
* Asynchronous response handling

## Installation

```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Install Python dependencies
pip install ollama

# For Linux TTS support (optional)
sudo apt-get install espeak  # Debian/Ubuntu
```

## Usage

```bash
# Basic usage (text-only)
python async-chat-stream_main.py

# Enable text-to-speech
python async-chat-stream_main.py --speak
```

### Interactive Commands

* Enter your message at the `>>>` prompt
* Press Ctrl+C or Ctrl+D to exit
* Empty input is ignored

## Code Structure

```python
async def speak(speaker, content):
    # Handles TTS output
    
async def main():
    # Main chat loop and program logic
    
# Core components:
# 1. Argument parsing
# 2. TTS system detection# 3. Async client initialization
# 4. Message history management
# 5. Streaming response handling
```

## Example Interaction

```
>>> Hello, how are you?
```


