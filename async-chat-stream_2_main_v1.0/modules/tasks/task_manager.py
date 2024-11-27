"""
Task management functionality for the async chat stream system.
"""
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
import json
from datetime import datetime
from ..utils.file_utils import sanitize_path, get_unique_filename
from ..config.constants import TASKS_FOLDER
from ..models.data_models import Task, TaskStatus, ChatMessage, MessageRole
from ..ui.menu_handler import Menu, MenuItem, FileSelector

logger = logging.getLogger(__name__)

class TaskManager:
    """Manages task creation, editing, and organization."""
    
    def __init__(self, tasks_dir: Union[str, Path]):
        self.tasks_dir = Path(tasks_dir)
        self.tasks_dir.mkdir(exist_ok=True)
        self.file_selector = FileSelector(self.tasks_dir, "*.json")
        
    def create_task_from_message(self, message: ChatMessage) -> Task:
        """Create a task from a chat message."""
        title = message.content.split('\n')[0][:50]  # First line, max 50 chars
        return Task(
            name=title,
            description=message.content,
            status=TaskStatus.PENDING,
            metadata={
                "source": "chat_conversion",
                "message_id": message.id,
                "timestamp": message.timestamp.isoformat()
            }
        )
    
    def save_tasks(self, tasks: List[Task], file_path: Path) -> None:
        """Save tasks to a JSON file."""
        with open(file_path, 'w') as f:
            json.dump([task.model_dump() for task in tasks], f, indent=2, default=str)
        logger.info(f"Saved {len(tasks)} tasks to {file_path}")
    
    def load_tasks(self, file_path: Path) -> List[Task]:
        """Load tasks from a JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        tasks = [Task(**task_data) for task_data in data]
        logger.info(f"Loaded {len(tasks)} tasks from {file_path}")
        return tasks
    
    def save_conversation_as_tasks(
        self,
        messages: List[ChatMessage],
        filename: str,
        append: bool = False
    ) -> Path:
        """Convert chat messages to tasks and save them."""
        if not messages:
            raise ValueError("No messages to convert")
            
        file_path = self.tasks_dir / f"{filename}.json"
        existing_tasks = []
        
        if append and file_path.exists():
            existing_tasks = self.load_tasks(file_path)
        
        new_tasks = []
        for i, msg in enumerate(messages):
            if msg.role == MessageRole.USER:
                task = self.create_task_from_message(msg)
                new_tasks.append(task)
            elif msg.role == MessageRole.ASSISTANT and new_tasks:
                # Add assistant's response to the last task's description
                new_tasks[-1].description += f"\n\nAssistant Response:\n{msg.content}"
        
        all_tasks = existing_tasks + new_tasks
        self.save_tasks(all_tasks, file_path)
        return file_path
    
    def edit_task_list(self, file_path: Optional[Path] = None) -> None:
        """Interactive task list editor."""
        if file_path is None:
            file_path = self.file_selector.select_file("Select task list to edit")
            if file_path is None:
                return
        
        tasks = self.load_tasks(file_path)
        while True:
            print(f"\n=== Task List: {file_path.stem} ===")
            for i, task in enumerate(tasks, 1):
                status = f"[{task.status.value}]"
                print(f"{i}. {status} {task.name}")
            
            menu = Menu("Task Actions", [
                MenuItem("1", "Add Task"),
                MenuItem("2", "Edit Task"),
                MenuItem("3", "Change Status"),
                MenuItem("4", "Delete Task"),
                MenuItem("5", "Save"),
                MenuItem("6", "Back")
            ])
            
            choice = menu.get_choice()
            if choice is None or choice.key == "6":
                break
                
            if choice.key == "1":  # Add Task
                name = input("Task name: ").strip()
                if name:
                    desc = input("Description (optional): ").strip()
                    task = Task(
                        name=name,
                        description=desc or name,
                        status=TaskStatus.PENDING
                    )
                    tasks.append(task)
                    
            elif choice.key == "2":  # Edit Task
                idx = input("Enter task number to edit: ").strip()
                if idx.isdigit() and 0 < int(idx) <= len(tasks):
                    task = tasks[int(idx) - 1]
                    name = input(f"New name ({task.name}): ").strip()
                    desc = input(f"New description ({task.description}): ").strip()
                    if name:
                        task.name = name
                    if desc:
                        task.description = desc
                        
            elif choice.key == "3":  # Change Status
                idx = input("Enter task number: ").strip()
                if idx.isdigit() and 0 < int(idx) <= len(tasks):
                    print("\nAvailable statuses:")
                    for i, status in enumerate(TaskStatus, 1):
                        print(f"{i}. {status.value}")
                    status_idx = input("Select status number: ").strip()
                    if status_idx.isdigit() and 0 < int(status_idx) <= len(TaskStatus):
                        tasks[int(idx) - 1].status = list(TaskStatus)[int(status_idx) - 1]
                        
            elif choice.key == "4":  # Delete Task
                idx = input("Enter task number to delete: ").strip()
                if idx.isdigit() and 0 < int(idx) <= len(tasks):
                    confirm = input(f"Delete task '{tasks[int(idx) - 1].name}'? (y/n): ").lower()
                    if confirm == 'y':
                        tasks.pop(int(idx) - 1)
                        
            elif choice.key == "5":  # Save
                self.save_tasks(tasks, file_path)
                print("Tasks saved successfully")
    
    def view_tasks(self, file_path: Optional[Path] = None) -> None:
        """View tasks with filtering and sorting options."""
        if file_path is None:
            file_path = self.file_selector.select_file("Select task list to view", allow_new=False)
            if file_path is None:
                return
        
        tasks = self.load_tasks(file_path)
        while True:
            menu = Menu("View Options", [
                MenuItem("1", "All Tasks"),
                MenuItem("2", "Pending Tasks"),
                MenuItem("3", "Completed Tasks"),
                MenuItem("4", "Sort by Status"),
                MenuItem("5", "Sort by Date"),
                MenuItem("6", "Back")
            ])
            
            choice = menu.get_choice()
            if choice is None or choice.key == "6":
                break
                
            filtered_tasks = tasks
            if choice.key == "2":
                filtered_tasks = [t for t in tasks if t.status == TaskStatus.PENDING]
            elif choice.key == "3":
                filtered_tasks = [t for t in tasks if t.status == TaskStatus.COMPLETED]
            elif choice.key == "4":
                filtered_tasks = sorted(tasks, key=lambda t: t.status.value)
            elif choice.key == "5":
                filtered_tasks = sorted(tasks, key=lambda t: t.created_at)
            
            print(f"\n=== {file_path.stem} ===")
            for i, task in enumerate(filtered_tasks, 1):
                status = f"[{task.status.value}]"
                created = task.created_at.strftime("%Y-%m-%d")
                print(f"{i}. {status} {task.name} (Created: {created})")
            
            input("\nPress Enter to continue...")
