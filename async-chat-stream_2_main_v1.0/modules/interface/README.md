# Interface Module

## Overview
The interface module demonstrates terminal-based user interface patterns and input handling in Python applications.

## Educational Value

### 1. Terminal UI
- Menu systems
- User prompts
- Input validation
- Display formatting

### 2. Input Processing
- Command parsing
- Input sanitization
- Error messaging
- Command history

### 3. State Management
- Menu navigation
- Session tracking
- Mode switching
- Exit handling

## Production Code Standards

### Type Safety
```python
from typing import Optional, List, Dict, Callable
from enum import Enum, auto
from dataclasses import dataclass

class Mode(Enum):
    """Application modes."""
    CHAT = auto()
    TASK = auto()
    CONFIG = auto()

@dataclass
class MenuItem:
    """Type-safe menu item."""
    key: str
    label: str
    handler: Callable
    help_text: Optional[str] = None

async def handle_input(
    prompt: str,
    validators: Optional[List[Callable]] = None
) -> str:
    """Handle user input with type safety."""
    pass
```

### Error Handling
```python
class InterfaceError(Exception):
    """Base interface exception."""
    pass

class InputValidationError(InterfaceError):
    """Input validation error."""
    pass

class ModeTransitionError(InterfaceError):
    """Mode transition error."""
    pass

async def safe_input_operation():
    """Safe input operation with error handling."""
    try:
        user_input = await handle_input("Enter command: ")
        if not user_input:
            raise InputValidationError("Empty input")
    except InterfaceError as e:
        logger.error(f"Interface error: {str(e)}")
        raise
    except Exception as e:
        logger.critical(f"Operation failed: {str(e)}")
        raise InterfaceError(str(e)) from e
```

### Input Validation
```python
from pydantic import BaseModel, Field

class CommandInput(BaseModel):
    """Validated command input."""
    command: str = Field(min_length=1)
    args: Optional[List[str]] = None
    
    @validator("command")
    def validate_command(cls, v):
        """Validate command format."""
        v = v.strip().lower()
        if not v in VALID_COMMANDS:
            raise ValueError(f"Invalid command: {v}")
        return v

def validate_input(value: str) -> str:
    """Validate and sanitize user input.
    
    Args:
        value: Raw input string
        
    Returns:
        Sanitized input
        
    Raises:
        InputValidationError: If input is invalid
    """
    # Remove control characters
    value = "".join(char for char in value if ord(char) >= 32)
    
    # Basic sanitization
    value = value.strip()
    
    # Validate length
    if not value:
        raise InputValidationError("Empty input")
    if len(value) > 1000:
        raise InputValidationError("Input too long")
        
    return value
```

### Logging
```python
logger = setup_logger(__name__, "interface.log")

async def interface_operation():
    """Interface operation with comprehensive logging."""
    session_id = str(uuid.uuid4())
    logger.info(f"Starting interface session: {session_id}")
    
    try:
        while True:
            try:
                cmd = await handle_input("Command: ")
                logger.debug(f"Received command: {cmd}")
                
                result = await process_command(cmd)
                logger.info(f"Command processed: {cmd}")
                
            except InputValidationError as e:
                logger.warning(f"Invalid input: {str(e)}")
                print(f"Error: {str(e)}")
                continue
                
    except Exception as e:
        logger.critical(f"Session {session_id} failed: {str(e)}")
        raise
    finally:
        logger.info(f"Ending session: {session_id}")
```

## Module Structure

### menu_manager.py
- Menu display
- Input handling
- Navigation logic
- Mode management

## Usage Example
```python
from modules.interface import MenuManager, Mode

async def interface_example():
    manager = MenuManager()
    
    try:
        # Display menu
        manager.display_menu()
        
        # Handle input
        cmd = await manager.get_input(
            prompt="Enter command: ",
            validators=[validate_command]
        )
        
        # Process command
        if cmd == "chat":
            await manager.switch_mode(Mode.CHAT)
        elif cmd == "exit":
            await manager.cleanup()
            
    except InterfaceError as e:
        logger.error(f"Interface error: {str(e)}")
        raise
```

## Best Practices

### Resource Management
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def interface_session():
    """Manage interface session."""
    manager = None
    try:
        manager = MenuManager()
        yield manager
    finally:
        if manager:
            await manager.cleanup()

async def run_interface():
    async with interface_session() as manager:
        await manager.run()
```

### Performance
- Command caching
- History management
- Efficient display updates
- Resource cleanup

### Security
- Input sanitization
- Command validation
- Mode transitions
- Session isolation

## Testing
```python
import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_menu_manager():
    """Test menu manager operations."""
    manager = MenuManager()
    
    # Test input handling
    mock_input = AsyncMock(return_value="chat")
    manager.get_input = mock_input
    
    cmd = await manager.get_input("Command: ")
    assert cmd == "chat"
    
    # Test mode switching
    await manager.switch_mode(Mode.CHAT)
    assert manager.current_mode == Mode.CHAT
```

## Core Functions

### Interface Operations
- `display_menu()`: Show menu options
- `get_input()`: Get validated input
- `switch_mode()`: Change system mode
- `process_command()`: Handle commands
