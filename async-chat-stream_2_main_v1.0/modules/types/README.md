# Types Module

## Overview

The types module showcases Python's type system and demonstrates how to create and use custom types for improved code safety and maintainability.

## Educational Value

### 1. Type System Features

* Custom type definitions
* Enums and dataclasses
* Type hints and annotations
* Optional and Union types

### 2. Object-Oriented Design

* Data encapsulation
* Immutable data structures
* Class inheritance
* Property decorators

### 3. Code Safety

* Type checking
* Runtime validation
* Error prevention
* IDE support

## Module Structure

### custom_types.py

* Model size categories (Enum)
* Chat message structure (Dataclass)
* Chat session management (Dataclass)
* Task definitions (TypedDict)

## Usage Example

```python
from modules.types.custom_types import (
    ModelSize,
    ChatMessage,
    ChatSession,
    OllamaModel
)

# Using enums for type safety
model_size = ModelSize.SMALL

# Creating typed messages
message = ChatMessage(
    role="user",
    content="Hello!",
    timestamp="2024-01-01T00:00:00"
)

# Type-safe session management
session = ChatSession(
    id="123",
    name="Test Chat",
    messages=[message],
    model_name="llama2",
    created_at="2024-01-01T00:00:00",
    updated_at="2024-01-01T00:00:00"
)
```

## Best Practices Demonstrated


1. Use of dataclasses for data containers
2. Enum for type-safe constants
3. Proper type annotations
4. Comprehensive docstrings
5. Immutable data structures where appropriate

## Type Definitions

### ModelSize

Enum representing different model size categories:

* EMBEDDING: Embedding models
* SMALL: < 5GB
* MEDIUM: 5GB - 20GB
* LARGE: > 20GB

### ChatMessage

Dataclass for chat messages:

* role: str
* content: str
* timestamp: str

### ChatSession

Dataclass for chat sessions:

* id: str
* name: Optional\[str\]
* messages: List\[ChatMessage\]
* model_name: str
* created_at: str
* updated_at: str


