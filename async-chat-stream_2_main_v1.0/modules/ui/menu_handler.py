"""
Menu handling system for the chat application.
Provides consistent menu display and input validation.
"""
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MenuOption(Enum):
    """Standard menu options across different menus."""
    CONTINUE = "Continue"
    SAVE = "Save"
    LOAD = "Load"
    NEW = "New"
    EDIT = "Edit"
    DELETE = "Delete"
    BACK = "Back"
    EXIT = "Exit"

@dataclass
class MenuItem:
    """Represents a menu item with its action."""
    key: str
    label: str
    action: Optional[Callable] = None
    description: Optional[str] = None

class Menu:
    """Handles menu display and selection."""
    def __init__(self, title: str, options: List[MenuItem]):
        self.title = title
        self.options = {opt.key: opt for opt in options}
        
    def display(self) -> None:
        """Display the menu with formatted options."""
        print(f"\n=== {self.title} ===")
        for key, item in self.options.items():
            desc = f" - {item.description}" if item.description else ""
            print(f"{key}. {item.label}{desc}")
            
    def get_choice(self, prompt: str = "Enter your choice") -> Optional[MenuItem]:
        """Get and validate user choice."""
        while True:
            try:
                choice = input(f"\n{prompt} ({', '.join(self.options.keys())}): ").strip()
                if choice.lower() in ['q', 'quit', 'exit']:
                    return None
                if choice in self.options:
                    return self.options[choice]
                print(f"Invalid choice. Please choose from: {', '.join(self.options.keys())}")
            except (KeyboardInterrupt, EOFError):
                return None

class FileSelector:
    """Handles file selection with pagination and search."""
    def __init__(self, directory: Path, pattern: str = "*"):
        self.directory = directory
        self.pattern = pattern
        self.page_size = 10
        
    def list_files(self, page: int = 1, search: Optional[str] = None) -> List[Path]:
        """List files with pagination and optional search."""
        files = sorted(self.directory.glob(self.pattern))
        if search:
            files = [f for f in files if search.lower() in f.stem.lower()]
        
        start = (page - 1) * self.page_size
        end = start + self.page_size
        return files[start:end]
        
    def select_file(self, prompt: str = "Select file", allow_new: bool = True) -> Optional[Path]:
        """Interactive file selection with search and pagination."""
        page = 1
        search = None
        
        while True:
            files = self.list_files(page, search)
            total_files = len(list(self.directory.glob(self.pattern)))
            total_pages = (total_files + self.page_size - 1) // self.page_size
            
            print(f"\n=== {prompt} (Page {page}/{total_pages}) ===")
            for i, file in enumerate(files, 1):
                print(f"{i}. {file.stem}")
                
            if allow_new:
                print("\n0. Create new file")
            print("n: Next page, p: Previous page, s: Search, q: Cancel")
            
            choice = input("\nEnter choice: ").strip().lower()
            
            if choice == 'q':
                return None
            elif choice == 'n' and page < total_pages:
                page += 1
            elif choice == 'p' and page > 1:
                page -= 1
            elif choice == 's':
                search = input("Enter search term: ").strip()
                page = 1
            elif choice == '0' and allow_new:
                name = input("Enter new file name: ").strip()
                if name:
                    return self.directory / f"{name}{self.pattern.replace('*', '')}"
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(files):
                    return files[idx]
            
            print("Invalid choice, please try again")
