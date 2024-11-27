"""
Main entry point for the async chat stream system.
"""
import asyncio
import logging
import ollama
from pathlib import Path
import uuid
from datetime import datetime

from modules.chat import process_chat_message, save_chat_session
from modules.models import get_available_models, filter_models
from modules.tasks.task_manager import TaskManager
from modules.ui.menu_handler import Menu, MenuItem
from modules.interface import (
    get_model_preferences,
    confirm_action,
    get_user_input
)
from modules.utils.logging_utils import setup_logger
from modules.types.custom_types import ChatSession, ChatMessage

# Set up logging
logger = setup_logger(__name__)

async def chat_mode(client: ollama.AsyncClient, model: str) -> None:
    """Interactive chat mode with the selected model."""
    logger.info(f"Starting chat mode with model: {model}")
    print(f"\nStarting chat with {model}")
    
    # Initialize task manager
    task_manager = TaskManager(Path("task_lists"))
    
    session = ChatSession(
        id=str(uuid.uuid4()),
        name=None,
        messages=[],
        model_name=model,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat()
    )
    
    while True:
        user_input = get_user_input("\nYou: ")
        if not user_input:
            continue
            
        session.messages.append(ChatMessage(
            role="user",
            content=user_input,
            timestamp=datetime.now().isoformat()
        ))
        
        try:
            response = await process_chat_message(
                client=client,
                model=model,
                messages=[{"role": m.role, "content": m.content} for m in session.messages]
            )
            
            session.messages.append(ChatMessage(
                role="assistant",
                content=response,
                timestamp=datetime.now().isoformat()
            ))
            
            print("\n")
            menu = Menu("Chat Actions", [
                MenuItem("1", "Continue Chatting"),
                MenuItem("2", "Save as Tasks"),
                MenuItem("3", "Save Chat History"),
                MenuItem("4", "Start New Chat"),
                MenuItem("5", "Edit Tasks"),
                MenuItem("6", "View Tasks"),
                MenuItem("7", "Return to Main Menu")
            ])
            
            choice = menu.get_choice()
            if not choice:
                break
                
            if choice.key == '1':  # Continue chatting
                continue
            elif choice.key == '2':  # Save as tasks
                filename = get_user_input("Enter task list name (without extension): ")
                if filename:
                    append = confirm_action("Append to existing list?")
                    task_manager.save_conversation_as_tasks(
                        messages=session.messages,
                        filename=filename,
                        append=append
                    )
            elif choice.key == '3':  # Save chat history
                if not session.name:
                    session.name = get_user_input("Enter chat name (press Enter for timestamp): ", allow_empty=True)
                save_chat_session(session)
            elif choice.key == '4':  # Start new chat
                if confirm_action("Start new chat? Current chat will be lost"):
                    session.messages = []
            elif choice.key == '5':  # Edit tasks
                task_manager.edit_task_list()
            elif choice.key == '6':  # View tasks
                task_manager.view_tasks()
            elif choice.key == '7':  # Return to main menu
                if session.messages and confirm_action("Save chat before exiting?"):
                    if not session.name:
                        session.name = get_user_input("Enter chat name: ", allow_empty=True)
                    save_chat_session(session)
                break
                
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            print(f"\nError: {str(e)}")
            if not confirm_action("Continue chatting?"):
                break

async def main():
    """Main execution function."""
    try:
        # Initialize task manager
        task_manager = TaskManager(Path("task_lists"))
        
        while True:
            menu = Menu("Main Menu", [
                MenuItem("1", "Start Chat Session"),
                MenuItem("2", "Manage Tasks"),
                MenuItem("3", "View Tasks"),
                MenuItem("4", "Exit")
            ])
            
            choice = menu.get_choice()
            if not choice:
                break
                
            if choice.key == "1":
                # Get model preferences and start chat
                size_pref, embedding = get_model_preferences()
                models = await get_available_models()
                filtered_models = filter_models(models, size_pref, embedding)
                
                if not filtered_models:
                    print("No models available matching preferences")
                    continue
                    
                print("\nAvailable Models:")
                model_menu = Menu("Select Model", [
                    MenuItem(str(i), model.name) 
                    for i, model in enumerate(filtered_models, 1)
                ])
                model_choice = model_menu.get_choice()
                
                if model_choice:
                    model_idx = int(model_choice.key) - 1
                    async with ollama.AsyncClient() as client:
                        await chat_mode(client, filtered_models[model_idx].name)
                        
            elif choice.key == "2":
                task_manager.edit_task_list()
            elif choice.key == "3":
                task_manager.view_tasks()
            elif choice.key == "4":
                print("Goodbye!")
                break
                
    except Exception as e:
        logger.error(f"Main execution error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)
