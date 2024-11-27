import logging
from typing import Dict, List, Optional, Any, Union, Tuple, Type
from pathlib import Path
import uuid
from datetime import datetime
import asyncio
import argparse
import ollama
import os
import time
import json
import csv
import re
import pandas as pd
from dataclasses import dataclass
from enum import Enum

# Add new constants
LOGS_FOLDER = "logs"
CHATS_FOLDER = "chats"
TASKS_FOLDER = "task_lists"

def get_project_root() -> Path:
    """
    Return the absolute path to the project root directory.
    
    Returns:
        Path: Absolute path to project root
        
    Raises:
        RuntimeError: If unable to determine project root
    """
    try:
        root = Path(__file__).parent.absolute()
        return root
    except Exception as e:
        print(f"Failed to determine project root: {str(e)}")
        raise RuntimeError(f"Could not determine project root: {str(e)}")

def create_log_folders() -> None:
    """Create necessary folders for logging."""
    try:
        root = get_project_root()
        logs_path = root / LOGS_FOLDER
        logs_path.mkdir(exist_ok=True)
        print(f"Created logs folder at: {logs_path}")
    except Exception as e:
        print(f"Failed to create logs folder: {str(e)}")
        raise

# Create logs folder before setting up logging
create_log_folders()

# Now set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(Path(get_project_root()) / LOGS_FOLDER / 'async_chat_stream.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def sanitize_path(path: Union[str, Path]) -> Path:
    """
    Sanitize and validate file path by resolving, checking for directory traversal attacks,
    and ensuring it's within project root.
    
    Args:
        path: Input path to sanitize
        
    Returns:
        Path: Sanitized absolute path object
        
    Raises:
        ValueError: If path contains invalid characters or attempts directory traversal
        OSError: If path resolution fails
    """
    try:
        # Convert to Path object and resolve
        clean_path = Path(path).resolve()
        
        # Prevent directory traversal
        if ".." in str(clean_path):
            raise ValueError("Directory traversal detected")
            
        # Make absolute if relative
        if not clean_path.is_absolute():
            clean_path = get_project_root() / clean_path
            
        logger.debug(f"Sanitized path from {path} to {clean_path}")
        return clean_path
        
    except ValueError as e:
        logger.error(f"Path validation failed: {path}, error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Path sanitization failed: {path}, error: {str(e)}")
        raise OSError(f"Failed to sanitize path: {str(e)}")

def reformat_csv(input_path: Union[str, Path], output_path: Union[str, Path]) -> None:
    """
    Reformat a CSV file by grouping and restructuring its contents.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to write reformatted output
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        PermissionError: If lacking write permissions
        ValueError: If CSV format is invalid
    """
    logger.info(f"Reformatting CSV file from {input_path} to {output_path}")
    
    try:
        input_path = sanitize_path(input_path)
        output_path = sanitize_path(output_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        current_group: Optional[str] = None
        
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
                
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    if '. ' in line and line.split('. ', 1)[0].isdigit():
                        parts = line.split('. ', 1)
                        if len(parts) == 2:
                            current_group = parts[1].split(' - ', 1)[0]
                            # Sanitize group name
                            current_group = re.sub(r'[^\w\s-]', '', current_group)
                            outfile.write(f"{current_group}:\n")
                    elif current_group:
                        if ' - ' in line:
                            task = line.split(' - ', 1)[-1]
                        else:
                            task = line
                        # Sanitize task
                        task = re.sub(r'[^\w\s-]', '', task)
                        outfile.write(f"- {task}\n")
                        
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {str(e)}")
                    continue
                    
        logger.info("CSV reformatting completed successfully")
        
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
        raise
    except PermissionError:
        logger.error(f"Permission denied writing to: {output_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to reformat CSV: {str(e)}")
        raise

def read_tasks(task_path: Union[str, Path]) -> Dict[str, List[str]]:
    """
    Read and parse tasks from a formatted text file into a dictionary structure.
    
    Args:
        task_path: Path to task file
        
    Returns:
        Dict[str, List[str]]: Dictionary mapping task groups to lists of tasks
        
    Raises:
        FileNotFoundError: If task file doesn't exist
        ValueError: If file format is invalid
    """
    logger.info(f"Reading tasks from: {task_path}")
    
    try:
        task_path = sanitize_path(task_path)
        tasks: Dict[str, List[str]] = {}
        current_group: Optional[str] = None

        with open(task_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    if line.endswith(':'):
                        current_group = line.rstrip(':')
                        # Sanitize group name
                        current_group = re.sub(r'[^\w\s-]', '', current_group)
                        tasks[current_group] = []
                    elif current_group and line.startswith('- '):
                        task = line[2:]
                        # Sanitize task
                        task = re.sub(r'[^\w\s-]', '', task)
                        tasks[current_group].append(task)
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {str(e)}")
                    continue

        if not tasks:
            logger.warning("No valid task groups found in file")
            return {}

        logger.info(f"Successfully loaded {len(tasks)} task groups")
        for group, task_list in tasks.items():
            logger.debug(f"Group '{group}' contains {len(task_list)} tasks")
            
        return tasks
        
    except FileNotFoundError:
        logger.error(f"Task file not found: {task_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to read tasks: {str(e)}")
        raise ValueError(f"Invalid task file format: {str(e)}")

# Hard-coded pre-prompt with input validation
PRE_PROMPT = """
You will be provided with a task, and you will be expected to complete only the given task at the time in context of the task above.
"""

# Hard-coded post-prompt with input validation  
POST_PROMPT = """
### The AI agent should assist in designing, developing, and refining the streaming radio station platform while adhering to these technical guidelines and coding practices. 
### The agent should provide guidance, code snippets, and architectural recommendations to ensure the application is built to a high standard.
### Remember to only complete the task at hand and not to do any other tasks. 
"""

async def speak(speaker: Optional[str], content: str) -> None:
    """
    Execute a text-to-speech command asynchronously.
    
    Args:
        speaker: Path to TTS executable
        content: Text content to speak
        
    Raises:
        RuntimeError: If speaker process fails
    """
    if speaker:
        try:
            logger.debug(f"Executing speaker command with content length: {len(content)}")
            p = await asyncio.create_subprocess_exec(speaker, content)
            await p.communicate()
            if p.returncode != 0:
                raise RuntimeError(f"Speaker process failed with code {p.returncode}")
        except Exception as e:
            logger.error(f"Failed to execute speaker: {str(e)}")
            raise

def create_output_folders(base_path: Union[str, Path], model_name: str, prompt_name: str) -> Dict[str, Path]:
    """
    Create organized output folder structure with unique naming and proper permissions.
    
    Args:
        base_path: Base directory for outputs
        model_name: Name of the model being used
        prompt_name: Name of the prompt category
        
    Returns:
        Dict[str, Path]: Dictionary of created folder paths
        
    Raises:
        OSError: If folder creation fails
        ValueError: If input parameters are invalid
    """
    try:
        # Validate inputs
        if not model_name or not prompt_name:
            raise ValueError("Model name and prompt name must not be empty")
            
        # Sanitize inputs
        model_name = re.sub(r'[^\w\s-]', '', model_name)
        prompt_name = re.sub(r'[^\w\s-]', '', prompt_name)
        
        base_path = sanitize_path(base_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = str(uuid.uuid4())[:8]
        
        model_path = base_path / 'model_tests' / model_name / f"{prompt_name}_{timestamp}_{run_id}"
        
        folders = {
            'txt_output': model_path / 'txt_output',
            'json_output': model_path / 'json_output',
            'js_output': model_path / 'js_output',
            'php_output': model_path / 'php_output',
            'python_output': model_path / 'python_output'
        }
        
        for folder_name, folder_path in folders.items():
            try:
                folder_path.mkdir(parents=True, exist_ok=True)
                os.chmod(folder_path, 0o755)  # rwxr-xr-x
                logger.info(f"Created output folder: {folder_path}")
            except Exception as e:
                logger.error(f"Failed to create/set permissions for {folder_name}: {str(e)}")
                raise
                
        return folders
        
    except ValueError as e:
        logger.error(f"Invalid input parameters: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Failed to create output folders: {str(e)}")
        raise OSError(f"Failed to create output folders: {str(e)}")

def log_metadata(file_path: Union[str, Path], start_time: datetime, end_time: datetime, token_count: int) -> Dict[str, Any]:
    """
    Log execution metadata for a processing run.
    
    Args:
        file_path: Path to processed file
        start_time: Processing start timestamp
        end_time: Processing end timestamp
        token_count: Number of tokens processed
        
    Returns:
        Dict[str, Any]: Metadata dictionary
        
    Raises:
        ValueError: If timestamps or token count are invalid
    """
    if end_time < start_time:
        raise ValueError("End time cannot be before start time")
    if token_count < 0:
        raise ValueError("Token count cannot be negative")
        
    try:
        metadata = {
            "file_path": str(sanitize_path(file_path)),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "token_count": token_count,
            "timestamp": datetime.now().isoformat()
        }
        logger.debug(f"Generated metadata: {metadata}")
        return metadata
    except Exception as e:
        logger.error(f"Failed to generate metadata: {str(e)}")
        raise

def get_unique_filename(file_path: Union[str, Path]) -> str:
    """
    Generate a unique filename by appending a version number if file exists.
    
    Args:
        file_path: Original file path
        
    Returns:
        str: Unique file path
        
    Raises:
        ValueError: If input path is invalid
    """
    try:
        file_path = sanitize_path(file_path)
        base, ext = os.path.splitext(str(file_path))
        counter = 1
        
        while os.path.exists(file_path):
            file_path = f"{base}_v{counter}{ext}"
            counter += 1
            if counter > 1000:  # Prevent infinite loops
                raise ValueError("Too many file versions")
                
        logger.debug(f"Generated unique filename: {file_path}")
        return str(file_path)
        
    except Exception as e:
        logger.error(f"Failed to generate unique filename: {str(e)}")
        raise

def create_language_folders(base_path: Union[str, Path], pre_prompt: str) -> List[str]:
    """
    Create output folders for detected programming languages in the prompt.
    
    Args:
        base_path: Base directory for language folders
        pre_prompt: Prompt text to scan for language references
        
    Returns:
        List[str]: List of detected languages
        
    Raises:
        OSError: If folder creation fails
        ValueError: If inputs are invalid
    """
    try:
        base_path = sanitize_path(base_path)
        
        if not pre_prompt:
            raise ValueError("Pre-prompt cannot be empty")
            
        # Find programming language mentions
        languages = re.findall(r'(?i)\b(javascript|php|python|html|css|sql)\b', pre_prompt)
        languages = list(set(lang.lower() for lang in languages))
        
        for lang in languages:
            try:
                folder_name = f"{lang}_output"
                folder_path = base_path / folder_name
                folder_path.mkdir(parents=True, exist_ok=True)
                os.chmod(folder_path, 0o755)
                logger.info(f"Created language folder: {folder_path}")
            except Exception as e:
                logger.error(f"Failed to create folder for {lang}: {str(e)}")
                raise
                
        return languages
        
    except ValueError as e:
        logger.error(f"Invalid input parameters: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Failed to create language folders: {str(e)}")
        raise OSError(f"Failed to create language folders: {str(e)}")

async def process_task(client, task, output_folders, languages, speaker, model):
    start_time = datetime.now()
    token_count = 0

    messages = [
        {'role': 'system', 'content': PRE_PROMPT},
        {'role': 'user', 'content': f"Task: {task}"},
        {'role': 'system', 'content': POST_PROMPT},
    ]
    
    content_out = ''
    message = {'role': 'assistant', 'content': ''}

    file_name = "_".join(task.split()[:5]).replace('/', '_') + f"_{start_time.strftime('%Y%m%d_%H%M%S')}"
    
    file_paths = {
        'json': get_unique_filename(os.path.join(output_folders['json_output'], f"{file_name}.json")),
        'txt': get_unique_filename(os.path.join(output_folders['txt_output'], f"{file_name}.txt")),
    }
    
    for lang in languages:
        lang_lower = lang.lower()
        if f'{lang_lower}_output' in output_folders:
            file_paths[lang_lower] = get_unique_filename(os.path.join(output_folders[f'{lang_lower}_output'], f"{file_name}.{lang_lower}"))

    with open(file_paths['txt'], "w") as txt_file, \
         open(file_paths['json'], "w") as json_file:
        
        lang_files = {lang.lower(): open(file_paths[lang.lower()], "w") for lang in languages if f'{lang.lower()}_output' in output_folders}
        
        try:
            async for response in await client.chat(model=model, messages=messages, stream=True):
                if response['done']:
                    messages.append(message)

                content = response['message']['content']
                txt_file.write(content)
                txt_file.flush()
                print(content, end='', flush=True)

                for lang in languages:
                    lang_lower = lang.lower()
                    if re.match(f'```{lang}', content, re.IGNORECASE):
                        lang_code = re.search(f'```{lang}\n(.*?)```', content, re.DOTALL | re.IGNORECASE)
                        if lang_code and lang_lower in lang_files:
                            lang_files[lang_lower].write(lang_code.group(1))
                            lang_files[lang_lower].flush()

                content_out += content
                if content in ['.', '!', '?', '\n']:
                    await speak(speaker, content_out)
                    content_out = ''

                message['content'] += content
                token_count += 1

            if content_out:
                await speak(speaker, content_out)
            print()

        finally:
            for file in lang_files.values():
                file.close()

    end_time = datetime.now()
    metadata = log_metadata(file_paths['txt'], start_time, end_time, token_count)
    
    with open(file_paths['txt'], "r+") as f:
        content = f.read()
        f.seek(0, 0)
        f.write(f"Task: {task}\n\nMetadata: {json.dumps(metadata, indent=2)}\n\n{content}")

    try:
        with open(file_paths['json'], "w") as f:
            json.dump({"task": task, "metadata": metadata, "content": message['content']}, f, indent=2)
    except Exception as e:
        print(f"Error saving JSON file: {e}")

    return metadata

class ModelSize(Enum):
    EMBEDDING = "embedding"  # Embedding models
    SMALL = "small"         # < 5GB
    MEDIUM = "medium"       # 5GB - 20GB
    LARGE = "large"        # > 20GB

@dataclass
class OllamaModel:
    name: str
    size: float  # in GB
    size_category: ModelSize
    is_embedding: bool = False

def is_embedding_model(model_name: str) -> bool:
    """
    Determine if a model is an embedding model based on its name.
    
    Args:
        model_name (str): Name of the model to check. Must be a non-empty string.
        
    Returns:
        bool: True if it's an embedding model, False otherwise
        
    Raises:
        ValueError: If model_name is empty or not a string
    """
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Model name must be a non-empty string")
        
    # Sanitize input by stripping whitespace and converting to lowercase
    model_name = model_name.strip().lower()
    
    embedding_keywords = [
        'embedding',
        'embed', 
        'text-embedding',
        'bge',
        'e5',
        'nomic-embed',
        'all-minilm'
    ]
    
    logger.debug(f"Checking if '{model_name}' is an embedding model")
    return any(keyword in model_name for keyword in embedding_keywords)

async def update_ollama_models_list(models_file_path: str) -> List[OllamaModel]:
    """
    Update and categorize the Ollama models list.
    
    Args:
        models_file_path (str): Path to the models list file
        
    Returns:
        List[OllamaModel]: List of categorized OllamaModel objects
    """
    if not isinstance(models_file_path, str) or not models_file_path.strip():
        raise ValueError("Models file path must be a non-empty string")
        
    logger.info(f"Updating Ollama models list at: {models_file_path}")
    
    try:
        # Validate Ollama connection
        client = ollama.AsyncClient(host='http://localhost:11434')
        response = await client.list()
        
        logger.debug(f"Retrieved models from Ollama")
        
        categorized_models: List[OllamaModel] = []
        embedding_models: List[Tuple[str, float]] = []
        chat_models: List[Tuple[str, float]] = []
        
        # Parse the response and categorize models
        for model in response['models']:
            # Extract model name from the tag field
            name = str(model['model']).strip()
            # Convert size from bytes to GB
            size_gb = float(model['size']) / (1024 * 1024 * 1024)  # Convert to GB
            
            if is_embedding_model(name):
                embedding_models.append((name, size_gb))
            else:
                chat_models.append((name, size_gb))
        
        logger.debug(f"Found {len(embedding_models)} embedding models and {len(chat_models)} chat models")
        
        # Process embedding models
        for name, size_gb in embedding_models:
            model_info = OllamaModel(
                name=name,
                size=size_gb,
                size_category=ModelSize.EMBEDDING,
                is_embedding=True
            )
            categorized_models.append(model_info)
        
        # Process chat models
        for name, size_gb in chat_models:
            size_cat = ModelSize.SMALL if size_gb < 5 else (
                ModelSize.MEDIUM if size_gb < 20 else ModelSize.LARGE
            )
            
            model_info = OllamaModel(
                name=name,
                size=size_gb,
                size_category=size_cat,
                is_embedding=False
            )
            categorized_models.append(model_info)
        
        # Save updated list to file
        try:
            with open(models_file_path, 'w') as f:
                f.write("Model Name,Size (GB),Category,Is Embedding\n")
                
                # Write embedding models first
                for model in categorized_models:
                    if model.is_embedding:
                        f.write(f"{model.name},{model.size:.2f},{model.size_category.value},True\n")
                
                # Write chat models grouped by size
                for size_cat in [ModelSize.SMALL, ModelSize.MEDIUM, ModelSize.LARGE]:
                    for model in categorized_models:
                        if not model.is_embedding and model.size_category == size_cat:
                            f.write(f"{model.name},{model.size:.2f},{model.size_category.value},False\n")
                            
            logger.info(f"Successfully wrote {len(categorized_models)} models to {models_file_path}")
            
        except IOError as e:
            logger.error(f"Failed to write models file: {str(e)}")
            raise IOError(f"Unable to write to models file: {str(e)}")
        
        return categorized_models
        
    except Exception as e:
        logger.error(f"Unexpected error updating models list: {str(e)}")
        raise

def select_model(models: List[OllamaModel], size_preference: Optional[ModelSize] = None, embedding_only: bool = False) -> Optional[str]:
    """
    Select a model based on size preference and type.
    
    Args:
        models (List[OllamaModel]): List of available models
        size_preference (Optional[ModelSize]): Preferred model size category
        embedding_only (bool): Whether to show only embedding models
        
    Returns:
        Optional[str]: Selected model name or None if cancelled
        
    Raises:
        ValueError: If models list is empty or invalid
    """
    if not isinstance(models, list):
        raise ValueError("Models must be provided as a list")
    
    if not models:
        logger.warning("Empty models list provided")
        return None
        
    logger.info(f"Selecting model with preferences - Size: {size_preference}, Embedding only: {embedding_only}")
    
    try:
        if embedding_only:
            filtered_models = [m for m in models if m.is_embedding]
        else:
            filtered_models = [m for m in models if not m.is_embedding]
            if size_preference:
                filtered_models = [m for m in filtered_models if m.size_category == size_preference]
        
        if not filtered_models:
            logger.info("No models found matching criteria")
            print("\nNo models found in this category.")
            return None
            
        # Display available models
        print("\nAvailable Models:")
        for i, model in enumerate(filtered_models, 1):
            print(f"{i}. {model.name} ({model.size:.2f}GB) - {model.size_category.value}")
        
        while True:
            try:
                choice = input("\nSelect model number (0 to cancel): ").strip()
                if not choice.isdigit():
                    print("Please enter a valid number")
                    continue
                    
                choice = int(choice)
                if choice == 0:
                    logger.info("Model selection cancelled by user")
                    return None
                if 1 <= choice <= len(filtered_models):
                    selected_model = filtered_models[choice-1].name
                    logger.info(f"Selected model: {selected_model}")
                    return selected_model
                print("Please enter a number within the valid range")
            except ValueError:
                print("Please enter a valid number")
                
    except Exception as e:
        logger.error(f"Error during model selection: {str(e)}")
        raise

def create_chat_folders() -> None:
    """Create necessary folders for chat storage."""
    try:
        root = get_project_root()
        chats_path = root / CHATS_FOLDER
        chats_path.mkdir(exist_ok=True)
        logger.info(f"Created chats folder at: {chats_path}")
    except Exception as e:
        logger.error(f"Failed to create chat folders: {str(e)}")
        raise

def save_chat_history(conversation: List[Dict[str, str]], chat_name: Optional[str] = None) -> str:
    """
    Save chat history to a JSON file.
    
    Args:
        conversation: List of conversation messages
        chat_name: Optional name for the chat file
        
    Returns:
        str: Path to saved chat file
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chat_name = chat_name or f"chat_{timestamp}"
        chat_name = re.sub(r'[^\w\s-]', '', chat_name)
        
        chat_path = get_project_root() / CHATS_FOLDER / f"{chat_name}.json"
        
        chat_data = {
            "timestamp": timestamp,
            "conversation": conversation,
            "metadata": {
                "messages_count": len(conversation),
                "last_updated": datetime.now().isoformat()
            }
        }
        
        with open(chat_path, 'w') as f:
            json.dump(chat_data, f, indent=2)
            
        logger.info(f"Saved chat history to: {chat_path}")
        return str(chat_path)
        
    except Exception as e:
        logger.error(f"Failed to save chat history: {str(e)}")
        raise

def load_chat_history(chat_name: str) -> List[Dict[str, str]]:
    """
    Load chat history from a JSON file.
    
    Args:
        chat_name: Name of the chat file to load
        
    Returns:
        List[Dict[str, str]]: Loaded conversation
    """
    try:
        chat_path = get_project_root() / CHATS_FOLDER / f"{chat_name}.json"
        if not chat_path.exists():
            raise FileNotFoundError(f"Chat file not found: {chat_path}")
            
        with open(chat_path, 'r') as f:
            chat_data = json.load(f)
            
        logger.info(f"Loaded chat history from: {chat_path}")
        return chat_data["conversation"]
        
    except Exception as e:
        logger.error(f"Failed to load chat history: {str(e)}")
        raise

def display_chat_menu() -> str:
    """Display chat action menu and get user choice."""
    print("\n=== Chat Actions ===")
    print("1. Continue chatting")
    print("2. Save to task list")
    print("3. Save chat history")
    print("4. Start new chat")
    print("5. Load previous chat")
    print("6. Edit task list")
    print("7. Return to main menu")
    
    while True:
        choice = input("\nEnter your choice (1-7): ").strip()
        if choice in ['1', '2', '3', '4', '5', '6', '7']:
            return choice
        print("Invalid choice. Please enter a number between 1 and 7.")

async def chat_mode(client: ollama.AsyncClient, model: str) -> None:
    """Interactive chat mode with the selected model."""
    if not isinstance(client, ollama.AsyncClient):
        raise ValueError("Invalid Ollama client")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Invalid model name")
        
    logger.info(f"Starting chat mode with model: {model}")
    print(f"\nStarting chat with {model}")
    
    conversation: List[Dict[str, str]] = []
    current_chat_name: Optional[str] = None
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if not user_input:
            continue
            
        conversation.append({"role": "user", "content": user_input})
        
        try:
            print("\nAssistant: ", end='', flush=True)
            response_content = ""
            
            async for part in await client.chat(
                model=model,
                messages=conversation,
                stream=True
            ):
                content = str(part['message']['content'])
                print(content, end='', flush=True)
                response_content += content
                
            conversation.append({"role": "assistant", "content": response_content})
            print("\n")
            
            # Display action menu after each exchange
            choice = display_chat_menu()
            
            if choice == '1':  # Continue chatting
                continue
            elif choice == '2':  # Save to task list
                existing_list = display_available_lists()
                if existing_list:
                    append = input("Append to existing list? (y/n): ").lower() == 'y'
                    save_conversation_as_tasks(conversation, existing_list, append)
                else:
                    save_path = input("Enter new task list name: ").strip()
                    if save_path:
                        save_conversation_as_tasks(conversation, save_path, False)
            elif choice == '3':  # Save chat history
                chats = list((get_project_root() / CHATS_FOLDER).glob('*.json'))
                if chats:
                    print("\nExisting chats:")
                    for i, chat in enumerate(chats, 1):
                        print(f"{i}. {chat.stem}")
                    append = input("\nAppend to existing chat? (y/n): ").lower() == 'y'
                    if append:
                        while True:
                            choice = input("Select chat number (0 for new chat): ").strip()
                            if choice == '0':
                                break
                            if choice.isdigit() and 0 < int(choice) <= len(chats):
                                existing_chat = load_chat_history(chats[int(choice)-1].stem)
                                conversation = existing_chat + conversation
                                current_chat_name = chats[int(choice)-1].stem
                                save_chat_history(conversation, current_chat_name)
                                print(f"Appended to chat: {current_chat_name}")
                                break
                            print("Invalid choice. Please try again.")
                    
                if not append:
                    chat_name = input("Enter chat name to save (press Enter for timestamp): ").strip()
                    current_chat_name = save_chat_history(conversation, chat_name)
                    print(f"Chat saved as: {current_chat_name}")
            elif choice == '4':  # Start new chat
                if conversation:
                    save = input("Save current chat before starting new? (y/n): ").lower()
                    if save == 'y':
                        chat_name = input("Enter chat name to save (press Enter for timestamp): ").strip()
                        save_chat_history(conversation, chat_name)
                conversation = []
                current_chat_name = None
                print("\nStarting new chat...")
            elif choice == '5':  # Load previous chat
                try:
                    chats = list((get_project_root() / CHATS_FOLDER).glob('*.json'))
                    if not chats:
                        print("No saved chats found.")
                        continue
                    print("\nAvailable chats:")
                    for i, chat in enumerate(chats, 1):
                        print(f"{i}. {chat.stem}")
                    choice = input("\nSelect chat number to load (0 to cancel): ").strip()
                    if choice.isdigit() and 0 < int(choice) <= len(chats):
                        conversation = load_chat_history(chats[int(choice)-1].stem)
                        current_chat_name = chats[int(choice)-1].stem
                        print(f"Loaded chat: {current_chat_name}")
                except Exception as e:
                    print(f"Error loading chat: {str(e)}")
            elif choice == '6':  # Edit task list
                view_current_tasks()
                # Add task list editing functionality here
            elif choice == '7':  # Return to main menu
                if conversation:
                    save = input("Save current chat before exiting? (y/n): ").lower()
                    if save == 'y':
                        chat_name = input("Enter chat name to save (press Enter for timestamp): ").strip()
                        save_chat_history(conversation, chat_name)
                logger.info("Exiting chat mode")
                break
                
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            print(f"\nError: {str(e)}")
            break

def get_available_lists() -> List[Path]:
    """Get list of available task lists."""
    try:
        task_path = get_project_root() / TASKS_FOLDER
        return sorted(task_path.glob('*.csv'))
    except Exception as e:
        logger.error(f"Failed to get task lists: {str(e)}")
        raise

def display_available_lists() -> Optional[str]:
    """Display available task lists and get user selection."""
    try:
        lists = get_available_lists()
        if not lists:
            print("No task lists found.")
            return None
            
        print("\nAvailable task lists:")
        for i, task_list in enumerate(lists, 1):
            print(f"{i}. {task_list.stem}")
            
        while True:
            choice = input("\nSelect list number (0 for new list): ").strip()
            if choice == '0':
                return None
            if choice.isdigit() and 0 < int(choice) <= len(lists):
                return lists[int(choice)-1].stem
            print("Invalid choice. Please try again.")
            
    except Exception as e:
        logger.error(f"Error displaying lists: {str(e)}")
        return None

def save_conversation_as_tasks(conversation: List[Dict[str, str]], filename: str, append: bool = False) -> None:
    """
    Save chat conversation as a task list.
    
    Args:
        conversation: List of conversation messages
        filename: Name for the task list file
        append: Whether to append to existing file
        
    Raises:
        ValueError: If conversation or filename is invalid
        IOError: If unable to write to file
    """
    if not isinstance(conversation, list) or not conversation:
        raise ValueError("Invalid or empty conversation")
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("Invalid filename")
        
    # Sanitize filename
    filename = "".join(c for c in filename if c.isalnum() or c in ('-', '_')).strip()
    if not filename:
        raise ValueError("Filename contains no valid characters")
        
    logger.info(f"{'Appending to' if append else 'Creating'} task list: {filename}")
    
    try:
        task_path = get_project_root() / TASKS_FOLDER / f"{filename}.csv"
        mode = 'a' if append else 'w'
        
        # Get last index from existing file if appending
        last_index = 0
        if append and task_path.exists():
            with open(task_path, 'r') as f:
                for line in f:
                    if line.strip() and line[0].isdigit():
                        try:
                            index = int(line.split('.', 1)[0])
                            last_index = max(last_index, index)
                        except ValueError:
                            continue
        
        with open(task_path, mode) as f:
            if append:
                f.write("\n")  # Add spacing between existing and new content
            for i, msg in enumerate(conversation, last_index + 1):
                if msg['role'] == 'user':
                    f.write(f"{i}. User Query - {msg['content']}\n")
                else:
                    f.write(f"{i}. Assistant Response - {msg['content']}\n")
                    
        logger.info(f"Successfully {'appended to' if append else 'saved'} {task_path}")
        print(f"\nConversation {'appended to' if append else 'saved to'} {task_path}")
        
    except IOError as e:
        logger.error(f"Failed to save conversation: {str(e)}")
        raise IOError(f"Unable to save conversation: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error saving conversation: {str(e)}")
        raise

def display_main_menu() -> str:
    """Display main menu and get user choice."""
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
            print(f"Error: {str(e)}")

def get_model_preferences() -> Tuple[Optional[ModelSize], bool]:
    """Get user preferences for model selection."""
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
                continue
            
        except Exception as e:
            print(f"Error: {str(e)}")

def view_current_tasks():
    """Display current tasks in the task list."""
    try:
        task_path = get_project_root() / 'task_lists'
        if not task_path.exists():
            print("\nNo tasks found.")
            return
            
        print("\n=== Current Task Lists ===")
        task_files = list(task_path.glob('*.csv'))
        
        if not task_files:
            print("No task lists found.")
            return
            
        for i, task_file in enumerate(task_files, 1):
            print(f"\n{i}. {task_file.name}")
            with open(task_file, 'r') as f:
                for line in f:
                    print(f"   {line.strip()}")
                    
    except Exception as e:
        logger.error(f"Error viewing tasks: {str(e)}")
        print(f"Error: {str(e)}")

async def main() -> None:
    """
    Main execution function with interactive menu.
    
    This function handles the main program flow including:
    - Loading and validating available models
    - Displaying and processing menu choices
    - Managing chat and task processing modes
    - Error handling and logging
    
    Returns:
        None
        
    Raises:
        FileNotFoundError: If models file cannot be found
        ConnectionError: If Ollama server connection fails
        ValueError: If invalid input is provided
        Exception: For other unexpected errors
    """
    logger.info("Starting main application")
    try:
        # Create necessary folders
        script_dir = get_project_root()
        create_log_folders()
        create_chat_folders()
        
        # Validate and sanitize models path relative to script directory
        models_path = script_dir / "ollama_models_list.txt"
        
        # Fetch initial models list
        print("Fetching available Ollama models...")
        try:
            # Test Ollama connection first
            client = ollama.AsyncClient(host='http://localhost:11434')
            await client.list()
            
            # Update models list
            models = await update_ollama_models_list(str(models_path))
            logger.info(f"Successfully loaded {len(models)} models")
            
            if not models:
                print("\nNo models found! Please ensure:")
                print("1. Ollama is running (run 'ollama serve' in a terminal)")
                print("2. You have models installed (run 'ollama pull model_name')")
                print("3. Try running 'ollama list' to verify your installation")
                return
                
        except ConnectionError:
            print("\nError: Could not connect to Ollama server!")
            print("Please ensure:")
            print("1. Ollama is installed (https://ollama.ai)")
            print("2. Ollama server is running (run 'ollama serve')")
            return
        except Exception as e:
            print(f"\nError initializing models: {str(e)}")
            return
        
        while True:
            try:
                # Get and validate menu choice
                choice = display_main_menu()
                logger.debug(f"User selected menu option: {choice}")
                
                if not isinstance(choice, str) or choice not in ['1','2','3','4','5']:
                    raise ValueError("Invalid menu selection")
                
                if choice == '5':  # Exit
                    logger.info("User requested exit")
                    print("\nExiting program...")
                    break
                    
                if choice == '3':  # View models
                    logger.debug("Entering model view mode")
                    size_pref, embedding_only = get_model_preferences()
                    selected = select_model(models, size_pref, embedding_only)
                    logger.debug(f"Model view complete. Selected: {selected}")
                    continue
                    
                if choice == '4':  # View tasks
                    logger.debug("Displaying task lists")
                    view_current_tasks()
                    continue
                    
                # Model selection for chat/task modes
                logger.info("Initiating model selection for chat/task mode")
                print("\nFirst, let's select a model:")
                size_pref, embedding_only = get_model_preferences()
                model_name = select_model(models, size_pref, embedding_only)
                
                if not model_name:
                    logger.warning("No model was selected")
                    print("No model selected. Returning to main menu.")
                    continue
                
                # Initialize Ollama client with connection validation
                try:
                    client = ollama.AsyncClient(host='http://localhost:11434')
                    # Test connection with a simple list request instead of health check
                    await client.list()
                    logger.info(f"Successfully connected to Ollama server with model: {model_name}")
                except Exception as e:
                    logger.error(f"Failed to connect to Ollama server: {str(e)}")
                    raise ConnectionError("Could not establish connection to Ollama server")
                
                if choice == '1':  # Chat mode
                    logger.info(f"Entering chat mode with model: {model_name}")
                    await chat_mode(client, model_name)
                elif choice == '2':  # Task processing
                    logger.info(f"Entering task processing mode with model: {model_name}")
                    # ... (keep existing task processing code)
                    pass

            except ValueError as ve:
                logger.error(f"Input validation error: {str(ve)}")
                print(f"Invalid input: {str(ve)}")
            except ConnectionError as ce:
                logger.error(f"Connection error: {str(ce)}")
                print(f"Connection failed: {str(ce)}")
                break

    except Exception as e:
        logger.error(f"Critical application error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        logger.info("Program terminated by user")
        print("\nExiting...")
    except Exception as e:
        logger.critical(f"Fatal error occurred: {str(e)}", exc_info=True)
        print(f"A fatal error occurred. Check logs for details.")