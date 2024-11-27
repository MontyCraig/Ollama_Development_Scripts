"""
User interface menu management for the async chat stream system.
"""
import logging
from typing import Optional, Tuple
from ..types.custom_types import ModelSize

logger = logging.getLogger(__name__)

def display_main_menu() -> str:
    """
    Display main menu and get user choice.
    
    Returns:
        str: Selected menu option
    """
    print("\n=== Async Chat Stream Processing System ===")
    print("1. Start Chat Session")
    print("2. Process Task List")
    print("3. View Available Models")
    print("4. View Current Tasks")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            if choice in ['1', '2', '3', '4', '5']:
                return choice
            print("Invalid choice. Please enter a number between 1 and 5.")
        except Exception as e:
            logger.error(f"Menu input error: {str(e)}")
            print("Invalid input. Please try again.")

def display_chat_menu() -> str:
    """
    Display chat action menu and get user choice.
    
    Returns:
        str: Selected menu option
    """
    print("\n=== Chat Actions ===")
    print("1. Continue chatting")
    print("2. Save to task list")
    print("3. Save chat history")
    print("4. Start new chat")
    print("5. Load previous chat")
    print("6. Edit task list")
    print("7. Return to main menu")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-7): ").strip()
            if choice in ['1', '2', '3', '4', '5', '6', '7']:
                return choice
            print("Invalid choice. Please enter a number between 1 and 7.")
        except Exception as e:
            logger.error(f"Menu input error: {str(e)}")
            print("Invalid input. Please try again.")

def get_model_preferences() -> Tuple[Optional[ModelSize], bool]:
    """
    Get user preferences for model selection.
    
    Returns:
        Tuple[Optional[ModelSize], bool]: Size preference and embedding flag
    """
    print("\n=== Model Preferences ===")
    print("Model Categories:")
    print("1. Small (< 5GB)")
    print("2. Medium (5GB - 20GB)")
    print("3. Large (> 20GB)")
    print("4. Embedding Models")
    print("5. Any Size (excluding embeddings)")
    
    while True:
        try:
            size_choice = input("\nSelect category (1-5): ").strip()
            if size_choice == '1':
                return ModelSize.SMALL, False
            elif size_choice == '2':
                return ModelSize.MEDIUM, False
            elif size_choice == '3':
                return ModelSize.LARGE, False
            elif size_choice == '4':
                return ModelSize.EMBEDDING, True
            elif size_choice == '5':
                return None, False
            else:
                print("Invalid choice. Please try again.")
        except Exception as e:
            logger.error(f"Model preference error: {str(e)}")
            print("Invalid input. Please try again.")

def confirm_action(prompt: str) -> bool:
    """
    Get user confirmation for an action.
    
    Args:
        prompt: Confirmation prompt text
        
    Returns:
        bool: True if confirmed, False otherwise
    """
    try:
        response = input(f"{prompt} (y/n): ").strip().lower()
        return response == 'y'
    except Exception as e:
        logger.error(f"Confirmation input error: {str(e)}")
        return False

def get_user_input(prompt: str, allow_empty: bool = False) -> Optional[str]:
    """
    Get validated user input.
    
    Args:
        prompt: Input prompt text
        allow_empty: Whether to allow empty input
        
    Returns:
        Optional[str]: User input or None if empty and allowed
    """
    try:
        value = input(prompt).strip()
        if not value and not allow_empty:
            print("Input cannot be empty. Please try again.")
            return None
        return value
    except Exception as e:
        logger.error(f"User input error: {str(e)}")
        return None
