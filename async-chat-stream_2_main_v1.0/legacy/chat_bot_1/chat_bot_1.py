from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
import logging
import os
import re
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import subprocess
import json
import tiktoken  # for token counting
import ollama
import sys

# Update import path - create_txt_report.py is now in the same directory as chat_bot_1.py
from create_txt_report import ChatReportGenerator

def sanitize_input(text: str) -> str:
    """
    Sanitize user input by removing special characters and limiting length.
    
    Args:
        text (str): Raw input text from user
        
    Returns:
        str: Sanitized text with special characters removed and length limited
    """
    # Remove special characters except basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    # Limit length to prevent injection
    return text[:1000]

def get_available_ollama_models() -> Dict[str, List[str]]:
    """
    Get list of available Ollama models installed on the system, organized by category.
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping categories to lists of model names
    """
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        
        # Create categorized model lists
        models = {
            'Code Models': [],
            'Chat Models': [],
            'Embedding Models': [],
            'Vision Models': [],
            'Other Models': []
        }
        
        for line in result.stdout.split('\n')[1:]:
            if not line.strip():
                continue
            
            model_name = line.split()[0]
            
            # Categorize models based on their names
            if any(x in model_name.lower() for x in ['code', 'coder', 'programming']):
                models['Code Models'].append(model_name)
            elif 'embed' in model_name.lower():
                models['Embedding Models'].append(model_name)
            elif any(x in model_name.lower() for x in ['vision', 'llava', 'moondream']):
                models['Vision Models'].append(model_name)
            elif any(x in model_name.lower() for x in ['chat', 'mistral', 'llama', 'phi']):
                models['Chat Models'].append(model_name)
            else:
                models['Other Models'].append(model_name)
                
        return models
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to get Ollama models: {str(e)}")
        return {}

def display_available_models(models: dict) -> None:
    """
    Display available models in a formatted way.
    
    Args:
        models (dict): Dictionary of categorized model lists
    """
    print("\nAvailable Ollama Models:")
    print("=" * 50)
    
    for category, model_list in models.items():
        if model_list:  # Only show categories with models
            print(f"\n{category}:")
            print("-" * len(category))
            # Sort and display models in columns
            model_list.sort()
            for i, model in enumerate(model_list, 1):
                print(f"{i:2d}. {model}")

def validate_model(model_name: str) -> Optional[str]:
    """
    Validate if requested model is available.
    
    Args:
        model_name (str): Name of the model to validate
        
    Returns:
        Optional[str]: Validated model name or None if invalid
    """
    available_models = get_available_ollama_models()
    all_models = [model for sublist in available_models.values() for model in sublist]
    
    if model_name in all_models:
        return model_name
    
    logging.warning(f"Model '{model_name}' not found.")
    display_available_models(available_models)
    return None

def setup_logging(log_dir: Path) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_dir (Path): Directory to store log files
        
    Returns:
        logging.Logger: Configured logger instance
        
    Raises:
        OSError: If log directory cannot be created
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(str(log_dir / 'chat_bot.log'))
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def initialize_chat():
    """
    Initialize chat bot with validated model.
    
    Returns:
        str: Selected model name
    """
    models = get_available_ollama_models()
    display_available_models(models)
    
    raise RuntimeError("No models found. Please install at least one model.")

class ChatBot:
    def __init__(self):
        # Use script directory as base for relative paths
        self.base_dir = Path(__file__).parent
        self.chats_dir = self.base_dir / "chats"
        self.chats_dir.mkdir(exist_ok=True)
        self.reports_dir = self.base_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        self.current_chat_file: Optional[Path] = None
        self.current_model: Optional[str] = None
        self.history: List[Dict] = []
        self.max_tokens_history = 2000
        self.report_generator = ChatReportGenerator(
            input_dir=str(self.chats_dir),
            output_dir=str(self.reports_dir)
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        try:
            encoder = tiktoken.get_encoding("cl100k_base")  # Default encoding
            return len(encoder.encode(text))
        except Exception as e:
            logging.warning(f"Token counting failed: {e}")
            return len(text.split()) * 2  # Rough estimation

    def get_truncated_history(self) -> List[Dict]:
        """Get history truncated to fit within token limit."""
        total_tokens = 0
        truncated_history = []
        
        for msg in reversed(self.history):
            msg_tokens = self.count_tokens(str(msg))
            if total_tokens + msg_tokens > self.max_tokens_history:
                break
            truncated_history.insert(0, msg)
            total_tokens += msg_tokens
            
        return truncated_history

    def load_chat_history(self, chat_file: Path) -> None:
        """Load chat history and model information from file."""
        try:
            with open(chat_file, 'r') as f:
                data = json.load(f)
                # Extract model information and history from JSON
                self.current_model = data.get('model')
                self.history = data.get('messages', [])
            self.current_chat_file = chat_file
            
            self.update_report()
            
            if not self.current_model:
                logging.warning("No model information found in chat history")
                # Prompt user to select a model if none found
                model_type, model_name = self.select_model()
                self.current_model = model_name
        except Exception as e:
            logging.error(f"Error loading chat history: {e}")
            self.history = []

    def save_chat_history(self) -> None:
        """Save chat history and model information to current file and update report."""
        if self.current_chat_file:
            try:
                data = {
                    'model': self.current_model,
                    'messages': self.history
                }
                # Save JSON
                with open(self.current_chat_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                # Update report
                self.update_report()
                
            except Exception as e:
                logging.error(f"Error saving chat history: {e}")

    def update_report(self) -> None:
        """Update the text report for the current chat."""
        try:
            if self.current_chat_file:
                # Process just this specific file
                self.report_generator.process_files()
        except Exception as e:
            logging.error(f"Error updating report: {e}")

    def get_available_chats(self) -> List[Path]:
        """Get list of available chat files."""
        return list(self.chats_dir.glob("*.json"))

    def create_new_chat(self, model_type: str) -> str:
        """Create new chat file with timestamp and model information."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_{model_type}_{timestamp}.json"
        self.current_chat_file = self.chats_dir / filename
        self.history = []
        # Initialize the file with model information
        self.save_chat_history()
        return filename

    def process_image_input(self) -> Optional[str]:
        """Process image input for vision models."""
        while True:
            image_path = input("Enter image URL or local path (or 'skip' to continue): ").strip()
            if image_path.lower() == 'skip':
                return None
            if image_path.startswith(('http://', 'https://')):
                return image_path
            path = Path(image_path)
            if path.exists() and path.is_file():
                return str(path.absolute())
            print("Invalid image path. Please try again.")

    def select_model(self) -> Tuple[str, str]:
        """
        Interactive model selection.
        
        Returns:
            Tuple[str, str]: Selected category and model name
        """
        models = get_available_ollama_models()
        
        if not models:
            raise RuntimeError("No Ollama models found. Please install at least one model.")
        
        # Display model categories
        print("\nAvailable model categories:")
        categories = list(models.keys())
        for i, category in enumerate(categories, 1):
            print(f"{i}. {category}")
        
        while True:
            try:
                category_idx = int(input("\nSelect category number: ")) - 1
                if 0 <= category_idx < len(categories):
                    category = categories[category_idx]
                    break
                print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Display models in selected category
        category_models = models[category]
        for i, model in enumerate(category_models, 1):
            print(f"{i}. {model}")
        
        while True:
            try:
                model_idx = int(input("\nSelect model number: ")) - 1
                if 0 <= model_idx < len(category_models):
                    return category, category_models[model_idx]
                print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    def chat_with_model(self) -> None:
        """Handle the main chat interaction with the selected Ollama model."""
        try:
            # Initialize Ollama client
            client = ollama.Client(host='http://localhost:11434')
            
            print(f"\nInitialized chat with model: {self.current_model}")
            print("Type 'exit' to end chat, 'image' for image input (if supported)")
            
            while True:
                try:
                    user_input = input("\nYou: ").strip()
                    
                    if user_input.lower() == 'exit':
                        # Final update before exiting
                        self.save_chat_history()
                        break
                    
                    if user_input.lower() == 'image':
                        image_path = self.process_image_input()
                        if not image_path:
                            continue
                        user_input = f"[Image: {image_path}] Please describe this image."
                    
                    # Sanitize input
                    user_input = sanitize_input(user_input)
                    
                    # Prepare messages including history context
                    messages = []
                    for msg in self.get_truncated_history():
                        messages.append({
                            'role': msg['role'],
                            'content': msg['content']
                        })
                    messages.append({
                        'role': 'user',
                        'content': user_input
                    })
                    
                    # Get response from model
                    response = client.chat(
                        model=self.current_model,
                        messages=messages,
                        stream=False
                    )
                    
                    bot_response = response['message']['content']
                    print(f"\nBot: {bot_response}")
                    
                    # Update history
                    self.history.append({
                        'role': 'user',
                        'content': user_input
                    })
                    self.history.append({
                        'role': 'assistant',
                        'content': bot_response
                    })
                    
                    # Update history and reports after each interaction
                    self.save_chat_history()
                    
                except Exception as e:
                    logging.error(f"Error in chat interaction: {str(e)}")
                    print("Sorry, there was an error processing your input. Please try again.")
                    
        except Exception as e:
            logging.error(f"Critical error in chat session: {str(e)}")
            print("Fatal error occurred. Please check logs for details.")
            raise

def main():
    """Main function to run the chat bot."""
    logging.info("Starting chat bot conversation")
    
    chatbot = ChatBot()
    
    # Option to continue previous chat or start new one
    available_chats = chatbot.get_available_chats()
    if available_chats:
        print("\nAvailable chat histories:")
        for i, chat_file in enumerate(available_chats, 1):
            try:
                with open(chat_file, 'r') as f:
                    data = json.load(f)
                    model = data.get('model', 'Unknown model')
                print(f"{i}. {chat_file.name} (Model: {model})")
            except Exception:
                print(f"{i}. {chat_file.name} (Error reading file)")
        print(f"{len(available_chats) + 1}. Start new chat")
        
        while True:
            try:
                choice = int(input("\nSelect option: ")) - 1
                if 0 <= choice < len(available_chats):
                    chatbot.load_chat_history(available_chats[choice])
                    break
                elif choice == len(available_chats):
                    break
                print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    
    # Select model if starting new chat
    if not chatbot.current_chat_file:
        model_type, model_name = chatbot.select_model()
        chatbot.current_model = model_name
        chatbot.create_new_chat(model_type)
    
    print(f"\nUsing model: {chatbot.current_model}")
    print("Type 'exit' to quit, 'image' to add image (for vision models)")
    
    # Start chat interaction
    chatbot.chat_with_model()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()