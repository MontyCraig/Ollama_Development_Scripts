# Ollama Chat Stream Processing System

A robust, modular Python framework for building advanced chat applications with Ollama models, featuring asynchronous processing, streaming responses, and comprehensive task management.

## 🌟 Features

* **Asynchronous Processing**: Built with Python's asyncio for efficient task handling
* **Streaming Responses**: Real-time streaming of model outputs
* **Multi-Model Support**: Compatible with various Ollama models
* **Task Management**: CSV-based task organization and tracking
* **Code Generation**: Support for multiple programming languages
* **Modular Design**: Clean, maintainable, and extensible architecture
* **Comprehensive Logging**: Detailed activity and performance tracking

## 🚀 Quick Start

### Prerequisites

* Python 3.8+
* Ollama installed and running
* Git (for version control)

### Installation

```bash
# Clone the repository
git clone https://github.com/MontyCraig/Ollama_Presentation.git
cd Ollama_Presentation

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run with default settings
python async-chat-stream_2_main_v1.0/main.py

# Enable text-to-speech
python async-chat-stream_2_main_v1.0/main.py --speak

# Use specific model
python async-chat-stream_2_main_v1.0/main.py --model qwen2:7b
```

## 🏗️ Project Structure

```
async-chat-stream_2_main_v1.0/
├── modules/           # Core functionality
├── legacy/           # Previous versions
└── docs/            # Documentation

chat_bot/
├── core/            # Core components
└── utils/           # Utility functions
```

## 📚 Documentation

* [Setup Guide](async-chat-stream_2_main_v1.0/SETUP.md)
* [Technical Report](async-chat-stream_2_main_v1.0/TECHNICAL_REPORT.md)
* [Code Standards](async-chat-stream_2_main_v1.0/CODING_STANDARDS.md)
* [Repository Structure](REPOSITORY_STRUCTURE.md)

## 🛠️ Features in Detail

### Task Management
* CSV-based task organization
* Interactive task selection
* Progress tracking
* Task completion persistence

### Output Management
* Structured file organization
* Multiple output formats
* Version control
* Metadata tracking

### Model Integration
* Asynchronous client implementation
* Streaming response handling
* Context management
* Token counting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

[Specify License]

## 🙏 Acknowledgments

* Ollama team for the excellent model framework
* Contributors and testers
* Open source community

## 📞 Contact

Monty Craig - [Your Contact Info]

Project Link: [https://github.com/MontyCraig/Ollama_Presentation](https://github.com/MontyCraig/Ollama_Presentation)
