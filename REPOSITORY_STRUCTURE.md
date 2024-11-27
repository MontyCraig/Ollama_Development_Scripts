# Repository Structure

## Overview

This repository contains the Ollama Chat Stream Processing System, organized into a modular structure for maintainability and scalability.

## Core Components

### ollama_models/

Configuration and specification files for Ollama models:

* `ollama_models_details.json`: Detailed model specifications
* `ollama_embeddings_models.json`: Embedding model capabilities
* `ollama_tool_use_models.json`: Tool-enabled models
* `ollama_vision_models.json`: Vision model specifications
* Model categorization and compatibility information

### async-chat-stream_2_main_v1.0/

Main application directory containing:

* `modules/`: Core functionality modules
* `legacy/`: Previous versions and implementations
* Documentation files (README.md, SETUP.md, etc.)

### chat_bot/

New modular framework containing:

* `core/`: Core components
  * `model_manager/`: Ollama model management
  * `test_framework/`: Testing infrastructure
  * `output_handler/`: Output processing
* `utils/`: Utility functions and helpers

### docs/

Project documentation:

* Technical guides
* Setup instructions
* API documentation
* Usage examples

### Reference Materials

The following materials are kept locally for reference but not tracked in git:

* `ollama-python/`: Original Ollama Python examples
* Legacy .gitignore files
* Example configurations

## File Organization

### Configuration

* Single root `.gitignore` for all ignore patterns
* Configuration files in respective module directories
* Environment-specific settings in separate files

### Documentation

* Module-level README files
* Technical documentation in `docs/`
* API documentation in respective modules

### Code Structure

* Modular design with clear separation of concerns
* Utility functions centralized in `utils/`
* Core functionality in dedicated modules

## Development Guidelines

### Version Control

* Main development in `main` branch
* Feature branches for new development
* Pull requests for code review
* Tag releases with semantic versioning

### Documentation

* Keep module-level documentation updated
* Document API changes
* Maintain changelog
* Update setup instructions

### Reference Materials

Reference materials and examples are kept locally but not tracked in git:

* Original implementations
* Example code
* Testing data
* Development notes


