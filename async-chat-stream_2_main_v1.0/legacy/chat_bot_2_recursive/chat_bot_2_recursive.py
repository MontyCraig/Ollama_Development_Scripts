from __future__ import annotations
import subprocess
import json
import ollama
from datetime import datetime
import os
import nltk
from nltk.tokenize import sent_tokenize
from pathlib import Path
import logging
from typing import Dict, List, Optional, Any, Union, NoReturn, Tuple
import re
import sys
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import asyncio
from functools import partial
import threading
from queue import Queue

class ChatHistory:
    def __init__(self, base_dir: Path):
        """Initialize chat history manager."""
        self.base_dir = base_dir
        self.json_dir = base_dir / 'json_outputs'
        self.convos_dir = base_dir / 'convos'
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.convos_dir.mkdir(parents=True, exist_ok=True)

    def save_chat(self, messages: List[Dict[str, str]], model_name: str, subject: str) -> Tuple[Path, Path]:
        """Save chat history in both JSON and text formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{model_name}_{subject}_{timestamp}"
        
        # Save JSON
        json_path = self.json_dir / f"{base_name}.json"
        with open(json_path, 'w') as f:
            json.dump(messages, f, indent=4)
            
        # Save text
        text_path = self.convos_dir / f"{base_name}.txt"
        with open(text_path, 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Subject: {subject}\n\n")
            
            for msg in messages:
                f.write(f"{msg['timestamp']} - {msg['role']}: {msg['content']}\n")
                
        return json_path, text_path

    def load_chat(self, filename: str) -> List[Dict[str, str]]:
        """Load chat history from JSON file."""
        try:
            with open(self.json_dir / filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading chat history: {str(e)}")

class ModelManager:
    def __init__(self, models_dir: Path):
        """Initialize model manager."""
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Define model type files
        self.model_files = {
            'general': self.models_dir / 'ollama_models.json',
            'code': self.models_dir / 'ollama_tool_use_models.json',
            'vision': self.models_dir / 'ollama_vision_models.json',
            'embedding': self.models_dir / 'ollama_embeddings_models.json'
        }

    def update_model_lists(self) -> Dict[str, List[Dict[str, str]]]:
        """Update all model list files."""
        models = self._get_ollama_models()
        
        # Categorize models
        categorized = {
            'general': [],
            'code': [],
            'vision': [],
            'embedding': []
        }
        
        for model in models:
            name = model['NAME'].lower()
            if 'code' in name or 'coder' in name:
                categorized['code'].append(model)
            elif 'vision' in name or 'visual' in name:
                categorized['vision'].append(model)
            elif 'embed' in name:
                categorized['embedding'].append(model)
            else:
                categorized['general'].append(model)
        
        # Save categorized models
        for category, model_list in categorized.items():
            with open(self.model_files[category], 'w') as f:
                json.dump(model_list, f, indent=4)
        
        return categorized

    def _get_ollama_models(self) -> List[Dict[str, str]]:
        """Get list of available Ollama models."""
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            check=True
        )
        
        models = []
        for line in result.stdout.splitlines()[1:]:  # Skip header
            if line.strip():
                parts = [p for p in line.split('  ') if p.strip()]
                if len(parts) >= 4:
                    models.append({
                        'NAME': parts[0].strip(),
                        'ID': parts[1].strip(),
                        'SIZE': parts[2].strip(),
                        'MODIFIED': ' '.join(parts[3:]).strip()
                    })
        
        return models

class ChatBotApp:
    def __init__(self, root: tk.Tk):
        """Initialize the ChatBot application."""
        self.root = root
        self.root.title("Ollama Chat Assistant")
        self.root.geometry("1200x800")
        
        # Initialize paths and managers
        self.base_path = Path(__file__).parent.absolute()
        self.models_dir = self.base_path / 'models'
        self.logs_dir = self.base_path / 'logs'
        
        self.model_manager = ModelManager(self.models_dir)
        self.chat_history = ChatHistory(self.base_path)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize UI
        self.setup_ui()
        
        # Load models
        self.load_models()
        
        # Message queue for async operations
        self.message_queue = Queue()
        self.setup_message_queue_handler()
        
        # Current chat state
        self.current_chat = []
        self.current_subject = None
        self.current_model = None

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for directory in [self.models_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def setup_logging(self) -> None:
        """Configure logging with rotating file handler."""
        log_file = self.logs_dir / 'chat_bot.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_ui(self) -> None:
        """Setup the main UI components."""
        # Main container with three panels
        self.main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for model selection
        self.setup_left_panel()
        
        # Middle panel for chat
        self.setup_middle_panel()
        
        # Right panel for chat history
        self.setup_right_panel()

    def setup_left_panel(self) -> None:
        """Setup the left panel with model selection."""
        left_frame = ttk.Frame(self.main_container)
        self.main_container.add(left_frame, weight=1)
        
        # Model type selection
        model_type_frame = ttk.Frame(left_frame)
        model_type_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(model_type_frame, text="Model Type:").pack(side=tk.LEFT)
        self.model_type = tk.StringVar(value="All Models")
        model_types = [
            "All Models",
            "Large Language Model (LLM)",
            "Code Generation",
            "Vision",
            "Embedding",
            "Tools",
            "Safety & Moderation",
            "Information Extraction"
        ]
        self.model_type_combo = ttk.Combobox(model_type_frame, textvariable=self.model_type, values=model_types, state="readonly")
        self.model_type_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.model_type_combo.bind('<<ComboboxSelected>>', self.on_model_type_changed)
        
        # Model list with columns
        self.model_tree = ttk.Treeview(left_frame, columns=('Name', 'Type', 'Size', 'Modified'), show='headings')
        self.model_tree.heading('Name', text='Model Name')
        self.model_tree.heading('Type', text='Type')
        self.model_tree.heading('Size', text='Size')
        self.model_tree.heading('Modified', text='Modified')
        
        # Set column widths
        self.model_tree.column('Name', width=250)
        self.model_tree.column('Type', width=150)
        self.model_tree.column('Size', width=80)
        self.model_tree.column('Modified', width=120)
        
        self.model_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.model_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.model_tree.configure(yscrollcommand=scrollbar.set)
        
        # Add copy button
        copy_button = ttk.Button(left_frame, text="Copy Model Name", command=self.copy_model_name)
        copy_button.pack(padx=5, pady=5)

    def copy_model_name(self) -> None:
        """Copy selected model name to clipboard."""
        selection = self.model_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a model to copy")
            return
            
        model_name = self.model_tree.item(selection[0])['values'][0]  # Name is first column
        self.root.clipboard_clear()
        self.root.clipboard_append(model_name)
        messagebox.showinfo("Copied", f"Model name '{model_name}' copied to clipboard")

    def on_model_type_changed(self, event=None) -> None:
        """Handle model type selection change."""
        self.display_filtered_models()

    def load_model_details(self) -> Dict[str, Dict[str, Any]]:
        """Load model details from the comprehensive JSON file."""
        details = {}
        try:
            details_file = self.models_dir / 'ollama_models_details.json'
            if details_file.exists():
                with open(details_file, 'r') as f:
                    data = json.load(f)
                    if 'models' in data:
                        for model in data['models']:
                            name = model.get('name', '')
                            if name:
                                details[name] = model
        except Exception as e:
            self.logger.warning(f"Error loading model details: {str(e)}")
        return details

    def display_filtered_models(self) -> None:
        """Display models filtered by selected type."""
        selected_type = self.model_type.get()
        
        # Clear current display
        self.model_tree.delete(*self.model_tree.get_children())
        
        try:
            # Get models and details
            models = self.get_ollama_list()
            details = self.load_model_details()
            
            # Filter based on selected type
            filtered_models = []
            for model in models:
                name = model['NAME']
                model_details = details.get(name, {})
                model_type = model_details.get('model_type', 'Unknown')
                
                if selected_type == "All Models" or model_type == selected_type:
                    filtered_models.append((name, model_details, model))
            
            # Display filtered models
            for name, details, model in filtered_models:
                model_type = details.get('model_type', 'Unknown')
                description = details.get('description', '')
                
                self.model_tree.insert('', 'end',
                                     values=(name, model_type, model['SIZE'], model['MODIFIED']),
                                     tags=(description,) if description else ())
                                     
            # Add tooltips for models with descriptions
            if hasattr(self.model_tree, 'tag_bind'):
                for item in self.model_tree.get_children():
                    tags = self.model_tree.item(item)['tags']
                    if tags:
                        tooltip_text = tags[0]
                        self.model_tree.tag_bind(tooltip_text, '<Enter>', 
                            lambda e, text=tooltip_text: self.show_tooltip(e, text))
                        self.model_tree.tag_bind(tooltip_text, '<Leave>', self.hide_tooltip)
                                     
        except Exception as e:
            self.logger.error(f"Error displaying filtered models: {str(e)}")
            messagebox.showerror("Error", f"Failed to display models: {str(e)}")

    def show_tooltip(self, event, text: str) -> None:
        """Show tooltip with model description."""
        x, y, _, _ = self.model_tree.bbox(self.model_tree.identify_row(event.y))
        x += self.model_tree.winfo_rootx() + 20  # Offset from cursor
        y += self.model_tree.winfo_rooty()
        
        self.tooltip = tk.Toplevel(self.root)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        
        # Create a frame with a border
        frame = ttk.Frame(self.tooltip, relief=tk.SOLID, borderwidth=1)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Add the description text with word wrap
        label = ttk.Label(frame, text=text, wraplength=400, justify=tk.LEFT,
                         background="#ffffe0", padding=(5, 5))
        label.pack(fill=tk.BOTH, expand=True)

    def hide_tooltip(self, event=None) -> None:
        """Hide the tooltip."""
        if hasattr(self, 'tooltip'):
            self.tooltip.destroy()
            self.tooltip = None

    def load_models(self) -> None:
        """Load and display models."""
        try:
            self.display_filtered_models()
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")

    def setup_middle_panel(self) -> None:
        """Setup the middle panel with chat interface."""
        chat_frame = ttk.Frame(self.main_container)
        self.main_container.add(chat_frame, weight=2)
        
        # Chat subject
        subject_frame = ttk.Frame(chat_frame)
        subject_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(subject_frame, text="Subject:").pack(side=tk.LEFT)
        self.subject_entry = ttk.Entry(subject_frame)
        self.subject_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Chat display
        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD)
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Input area
        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.input_field = ttk.Entry(input_frame)
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(input_frame, text="Send", command=self.send_message).pack(side=tk.RIGHT, padx=5)
        ttk.Button(input_frame, text="Clear", command=self.clear_chat).pack(side=tk.RIGHT)
        
        self.input_field.bind('<Return>', lambda e: self.send_message())

    def setup_right_panel(self) -> None:
        """Setup the right panel with chat history."""
        history_frame = ttk.Frame(self.main_container)
        self.main_container.add(history_frame, weight=1)
        
        # History header
        ttk.Label(history_frame, text="Chat History").pack(padx=5, pady=5)
        
        # History list
        self.history_tree = ttk.Treeview(history_frame, columns=('Date', 'Model'), show='headings')
        self.history_tree.heading('Date', text='Date')
        self.history_tree.heading('Model', text='Model')
        self.history_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # History controls
        controls_frame = ttk.Frame(history_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(controls_frame, text="Load", command=self.load_selected_chat).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Delete", command=self.delete_selected_chat).pack(side=tk.LEFT)

    def setup_message_queue_handler(self) -> None:
        """Setup message queue handler for async operations."""
        def check_queue():
            while True:
                try:
                    message = self.message_queue.get_nowait()
                    self.chat_display.insert(tk.END, message + '\n')
                    self.chat_display.see(tk.END)
                except:
                    break
            self.root.after(100, check_queue)
        self.root.after(100, check_queue)

    def get_ollama_list(self) -> List[Dict[str, str]]:
        """Fetch list of available Ollama models."""
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                check=True,
                timeout=30
            )
            
            lines = [line.strip() for line in result.stdout.split('\n') if line.strip()]
            if not lines:
                return []
                
            models = []
            for line in lines[1:]:  # Skip header
                parts = [part for part in line.split('  ') if part.strip()]
                if len(parts) >= 4:
                    model = {
                        'NAME': parts[0].strip(),
                        'ID': parts[1].strip(),
                        'SIZE': parts[2].strip(),
                        'MODIFIED': ' '.join(parts[3:]).strip()
                    }
                    models.append(model)
            
            # Save to models directory
            models_file = self.models_dir / 'ollama_models.json'
            with open(models_file, 'w') as f:
                json.dump(models, f, indent=4)
            
            return models
            
        except Exception as e:
            self.logger.error(f"Error getting Ollama models: {str(e)}")
            raise

    def send_message(self) -> None:
        """Handle sending a message."""
        message = self.input_field.get().strip()
        if not message:
            return
        
        # Check subject
        if not self.current_subject:
            subject = self.subject_entry.get().strip()
            if not subject:
                messagebox.showerror("Error", "Please enter a subject for the chat")
                return
            self.current_subject = subject
            self.subject_entry.configure(state='disabled')
        
        # Get selected model
        if not self.current_model:
            selected = self.get_selected_model()
            if not selected:
                messagebox.showerror("Error", "Please select a model first")
                return
            self.current_model = selected
        
        # Clear input
        self.input_field.delete(0, tk.END)
        
        # Add message to chat
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.current_chat.append({
            'timestamp': timestamp,
            'role': 'user',
            'content': message
        })
        
        self.chat_display.insert(tk.END, f"You ({timestamp}): {message}\n")
        
        # Start async chat
        threading.Thread(
            target=lambda: asyncio.run(self.handle_chat(message)),
            daemon=True
        ).start()

    async def handle_chat(self, message: str) -> None:
        """Handle chat interaction with selected model."""
        try:
            client = ollama.AsyncClient()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            self.chat_display.insert(tk.END, f"AI ({timestamp}): ")
            content_out = ''
            
            async for response in await client.chat(
                model=self.current_model,
                messages=[{'role': 'user', 'content': message}],
                stream=True
            ):
                if 'message' in response and 'content' in response['message']:
                    content = response['message']['content']
                    self.chat_display.insert(tk.END, content)
                    self.chat_display.see(tk.END)
                    content_out += content
            
            self.chat_display.insert(tk.END, '\n')
            
            # Add response to chat history
            self.current_chat.append({
                'timestamp': timestamp,
                'role': 'assistant',
                'content': content_out
            })
            
            # Save chat
            self.save_current_chat()
            
        except Exception as e:
            self.logger.error(f"Error in chat: {str(e)}")
            self.chat_display.insert(tk.END, f"Error: {str(e)}\n")

    def save_current_chat(self) -> None:
        """Save the current chat session."""
        try:
            self.chat_history.save_chat(
                self.current_chat,
                self.current_model,
                self.current_subject
            )
            self.update_history_display()
        except Exception as e:
            self.logger.error(f"Error saving chat: {str(e)}")
            messagebox.showerror("Error", f"Failed to save chat: {str(e)}")

    def clear_chat(self) -> None:
        """Clear the current chat session."""
        if self.current_chat and messagebox.askyesno(
            "Clear Chat",
            "Do you want to save the current chat before clearing?"
        ):
            self.save_current_chat()
        
        self.current_chat = []
        self.current_subject = None
        self.current_model = None
        self.chat_display.delete('1.0', tk.END)
        self.subject_entry.configure(state='normal')
        self.subject_entry.delete(0, tk.END)

    def get_selected_model(self) -> Optional[str]:
        """Get the currently selected model name."""
        selected_tab = self.model_notebook.select()
        if not selected_tab:
            return None
            
        tree = self.model_notebook.children[selected_tab].winfo_children()[0]
        selection = tree.selection()
        
        if not selection:
            return None
            
        return tree.item(selection[0])['text']

    def load_selected_chat(self) -> None:
        """Load the selected chat history."""
        selection = self.history_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a chat to load")
            return
            
        try:
            item = self.history_tree.item(selection[0])
            chat_file = item['values'][0]  # Assuming filename is stored in first column
            
            # Load chat history
            chat_data = self.chat_history.load_chat(chat_file)
            
            # Clear current chat
            self.clear_chat()
            
            # Display loaded chat
            for message in chat_data:
                timestamp = message.get('timestamp', '')
                role = message.get('role', '')
                content = message.get('content', '')
                self.chat_display.insert(tk.END, f"{role} ({timestamp}): {content}\n")
                
            # Update subject if available
            if chat_data and 'subject' in chat_data[0]:
                self.subject_entry.delete(0, tk.END)
                self.subject_entry.insert(0, chat_data[0]['subject'])
                
        except Exception as e:
            self.logger.error(f"Error loading chat: {str(e)}")
            messagebox.showerror("Error", f"Failed to load chat: {str(e)}")

    def delete_selected_chat(self) -> None:
        """Delete the selected chat history."""
        selection = self.history_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a chat to delete")
            return
            
        if not messagebox.askyesno("Confirm Delete", "Are you sure you want to delete this chat?"):
            return
            
        try:
            item = self.history_tree.item(selection[0])
            chat_file = item['values'][0]  # Assuming filename is stored in first column
            
            # Delete the files
            json_path = self.chat_history.json_dir / chat_file
            text_path = self.chat_history.convos_dir / chat_file.replace('.json', '.txt')
            
            if json_path.exists():
                json_path.unlink()
            if text_path.exists():
                text_path.unlink()
                
            # Remove from tree
            self.history_tree.delete(selection[0])
            
        except Exception as e:
            self.logger.error(f"Error deleting chat: {str(e)}")
            messagebox.showerror("Error", f"Failed to delete chat: {str(e)}")

    def update_history_display(self) -> None:
        """Update the chat history display."""
        # Clear current items
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
            
        # List all JSON files in the history directory
        try:
            for file in self.chat_history.json_dir.glob('*.json'):
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                        if data:
                            # Get metadata from first message
                            timestamp = data[0].get('timestamp', 'Unknown')
                            model = data[0].get('model', 'Unknown')
                            self.history_tree.insert('', 'end', values=(file.name, timestamp, model))
                except Exception as e:
                    self.logger.warning(f"Error reading chat file {file}: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Error updating history display: {str(e)}")

def main():
    root = tk.Tk()
    app = ChatBotApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()