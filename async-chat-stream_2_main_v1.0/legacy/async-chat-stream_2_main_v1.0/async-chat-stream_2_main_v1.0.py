"""
Main script for Async Chat Stream Processing System.
"""
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
import re
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field, validator
import csv

# Define enums and models that were previously imported
class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user" 
    ASSISTANT = "assistant"

class ModelSize(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ChatMessage:
    role: MessageRole
    content: str
    timestamp: datetime = datetime.now()

@dataclass 
class ChatSession:
    id: str
    messages: List[ChatMessage]
    model: str
    created_at: datetime = datetime.now()

class OllamaModel(BaseModel):
    name: str
    size: ModelSize
    description: str = ""
    parameters: Dict[str, Any] = Field(default_factory=dict)

class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

# Add new constants - relative to script directory
LOGS_FOLDER = "logs"
CHATS_FOLDER = "chats" 
TASKS_FOLDER = "task_lists"

def get_script_dir() -> Path:
    """
    Return the absolute path to the script's directory.
    
    Returns:
        Path: Absolute path to script directory
        
    Raises:
        RuntimeError: If unable to determine script directory
    """
    try:
        return Path(__file__).resolve().parent
    except Exception as e:
        print(f"Failed to determine script directory: {str(e)}")
        raise RuntimeError(f"Could not determine script directory: {str(e)}")

def create_log_folders() -> None:
    """Create necessary folders for logging relative to script."""
    try:
        script_dir = get_script_dir()
        logs_path = script_dir / LOGS_FOLDER
        logs_path.mkdir(exist_ok=True)
        print(f"Created logs folder at: {logs_path}")
    except Exception as e:
        print(f"Failed to create logs folder: {str(e)}")
        raise

# Create logs folder before setting up logging
create_log_folders()

# Now set up logging relative to script directory
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(Path(get_script_dir()) / LOGS_FOLDER / 'async_chat_stream.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def sanitize_path(path: Union[str, Path]) -> Path:
    """
    Sanitize and validate file path by resolving, checking for directory traversal attacks,
    and ensuring it's within script directory.
    
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
            
        # Make absolute relative to script directory if relative
        if not clean_path.is_absolute():
            clean_path = get_script_dir() / clean_path
            
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
def get_user_prompts() -> Tuple[str, str]:
    """
    Get pre-prompt and post-prompt from user input.
    
    Returns:
        Tuple[str, str]: Pre-prompt and post-prompt strings
        
    Raises:
        ValueError: If prompts are empty
    """
    print("\nPlease enter the pre-prompt that will be used for all tasks.")
    print("This sets up the initial context for the AI. Leave blank for default.")
    pre_prompt = input("> ").strip()
    
    if not pre_prompt:
        pre_prompt = """
        You will be provided with a task, and you will be expected to complete only the given task at the time in context of the task above.
        """
        print(f"Using default pre-prompt: {pre_prompt}")
        
    print("\nPlease enter the post-prompt that will be used after each task.")
    print("This provides final instructions. Leave blank for default.")
    post_prompt = input("> ").strip()
    
    if not post_prompt:
        post_prompt = """
        ### The AI agent should assist in designing, developing, and refining the streaming radio station platform while adhering to these technical guidelines and coding practices. 
        ### The agent should provide guidance, code snippets, and architectural recommendations to ensure the application is built to a high standard.
        ### Remember to only complete the task at hand and not to do any other tasks.
        """
        print(f"Using default post-prompt: {post_prompt}")
        
    if not pre_prompt or not post_prompt:
        raise ValueError("Prompts cannot be empty")
        
    return pre_prompt, post_prompt

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
        
        # Make paths relative to script directory
        script_dir = get_script_dir()
        base_path = script_dir / sanitize_path(base_path)
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
        base_path = Path(base_path).resolve()
        
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
    # Get prompts from user
    pre_prompt, post_prompt = get_user_prompts()
    
    start_time = datetime.now()
    token_count = 0

    messages = [
        {'role': 'system', 'content': pre_prompt},
        {'role': 'user', 'content': f"Task: {task}"},
        {'role': 'system', 'content': post_prompt},
    ]
    
    content_out = ''
    message = {'role': 'assistant', 'content': ''}

    file_name = "_".join(task.split()[:5]).replace('/', '_') + f"_{start_time.strftime('%Y%m%d_%H%M%S')}"
    
    file_paths = {
        'json': get_unique_filename(Path(output_folders['json_output']) / f"{file_name}.json"),
        'txt': get_unique_filename(Path(output_folders['txt_output']) / f"{file_name}.txt"),
    }
    
    for lang in languages:
        lang_lower = lang.lower()
        if f'{lang_lower}_output' in output_folders:
            file_paths[lang_lower] = get_unique_filename(Path(output_folders[f'{lang_lower}_output']) / f"{file_name}.{lang_lower}")

    with open(file_paths['txt'], "w") as txt_file, \
         open(file_paths['json'], "w") as json_file:
        
        lang_files = {lang.lower(): open(file_paths[lang_lower], "w") for lang in languages if f'{lang_lower}_output' in output_folders}
        
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

async def fetch_ollama_models() -> list[dict]:
    """Fetch available models from Ollama CLI."""
    try:
        process = await asyncio.create_subprocess_exec(
            'ollama', 'list',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logging.error(f"Error fetching models: {stderr.decode()}")
            return []
            
        # Parse the output and convert to list of dicts
        output = stdout.decode().strip().split('\n')[1:]  # Skip header
        models = []
        for line in output:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                model_name = parts[0]
                size = parts[1]
                models.append({
                    "name": model_name,
                    "size": size,
                    "category": "small" if float(size.replace('GB', '')) < 5 else "medium" if float(size.replace('GB', '')) < 15 else "large",
                    "is_embedding": False
                })
        return models
    except Exception as e:
        logging.error(f"Error fetching models: {str(e)}")
        return []

async def save_models_to_file(models: list[dict], filepath: str) -> None:
    """Save models list to CSV file."""
    try:
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Model Name", "Size (GB)", "Category", "Is Embedding"])
            for model in models:
                writer.writerow([
                    model["name"],
                    model["size"],
                    model["category"],
                    model["is_embedding"]
                ])
    except Exception as e:
        logging.error(f"Error saving models to file: {str(e)}")

async def update_ollama_models_list(filepath: str) -> list[str]:
    """Update and load the Ollama models list."""
    models = await fetch_ollama_models()
    if models:
        await save_models_to_file(models, filepath)
        return [model["name"] for model in models]
    return []

async def process_task(client, task, output_folders, languages, speaker, model):
    # Get prompts from user
    pre_prompt, post_prompt = get_user_prompts()
    
    start_time = datetime.now()
    token_count = 0

    messages = [
        {'role': 'system', 'content': pre_prompt},
        {'role': 'user', 'content': f"Task: {task}"},
        {'role': 'system', 'content': post_prompt},
    ]
    
    content_out = ''
    message = {'role': 'assistant', 'content': ''}

    file_name = "_".join(task.split()[:5]).replace('/', '_') + f"_{start_time.strftime('%Y%m%d_%H%M%S')}"
    
    file_paths = {
        'json': get_unique_filename(Path(output_folders['json_output']) / f"{file_name}.json"),
        'txt': get_unique_filename(Path(output_folders['txt_output']) / f"{file_name}.txt"),
    }
    
    for lang in languages:
        lang_lower = lang.lower()
        if f'{lang_lower}_output' in output_folders:
            file_paths[lang_lower] = get_unique_filename(Path(output_folders[f'{lang_lower}_output']) / f"{file_name}.{lang_lower}")

    with open(file_paths['txt'], "w") as txt_file, \
         open(file_paths['json'], "w") as json_file:
        
        lang_files = {lang.lower(): open(file_paths[lang_lower], "w") for lang in languages if f'{lang_lower}_output' in output_folders}
        
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

async def update_ollama_models_list(models_file_path: str) -> List[OllamaModel]:
    """
    Update and categorize the Ollama models list.
    
    Args:
        models_file_path (str): Path to the models list file. Must be a valid file path.
        
    Returns:
        List[OllamaModel]: List of categorized OllamaModel objects
        
    Raises:
        ValueError: If models_file_path is invalid
        ConnectionError: If unable to connect to Ollama server
        IOError: If unable to write to models file
    """
    try:
        models_file_path = Path(models_file_path).resolve()
        
        # Get list of models from Ollama API
        client = ollama.AsyncClient()
        models_response = await client.list()
        
        if not models_response or not hasattr(models_response, 'models'):
            logger.error(f"Invalid response from Ollama API: {models_response}")
            raise ValueError("Invalid response from Ollama API")
            
        # Parse and categorize models
        models_list = []
        for model in models_response.models:
            # Extract model info from the new response structure
            name = model.model
            if not name:
                logger.warning("Found model with empty name, skipping")
                continue
                
            # Size is now in bytes, convert to GB
            size_gb = float(model.size) / (1024 * 1024 * 1024)  # Convert to GB
            
            # Determine if it's an embedding model
            is_embedding = is_embedding_model(name)
            
            # Categorize by size
            if is_embedding:
                size_category = ModelSize.EMBEDDING
            elif size_gb < 5:
                size_category = ModelSize.SMALL
            elif size_gb <= 20:
                size_category = ModelSize.MEDIUM
            else:
                size_category = ModelSize.LARGE
                
            # Create OllamaModel object
            model_obj = OllamaModel(
                name=name,
                size=size_gb,
                size_category=size_category,
                is_embedding=is_embedding
            )
            models_list.append(model_obj)
            
        # Sort models by size
        models_list.sort(key=lambda x: x.size)
        
        # Save to file for caching
        with open(models_file_path, 'w') as f:
            json.dump([{
                'name': m.name,
                'size': m.size,
                'size_category': m.size_category.value,
                'is_embedding': m.is_embedding
            } for m in models_list], f, indent=2)
            
        logger.info(f"Successfully processed {len(models_list)} models")
        return models_list
        
    except Exception as e:
        logger.error(f"Error updating models list: {str(e)}")
        raise

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
    if not isinstance(model_name, str):
        logger.warning(f"Invalid model name type: {type(model_name)}")
        return False
        
    if not model_name.strip():
        logger.warning("Empty model name provided")
        return False
        
    embedding_keywords = [
        'embedding',
        'embed',
        'text-embedding',
        'all-minilm',
        'bge-',
        'nomic-embed',
        'e5-'
    ]
    
    logger.debug(f"Checking if '{model_name}' is an embedding model")
    return any(keyword in model_name.lower() for keyword in embedding_keywords)

def create_chat_folders() -> None:
    """Create necessary folders for chat storage."""
    try:
        script_dir = Path(__file__).parent.resolve()
        chats_path = script_dir / CHATS_FOLDER
        tasks_path = script_dir / TASKS_FOLDER
        chats_path.mkdir(exist_ok=True)
        tasks_path.mkdir(exist_ok=True)
        print(f"Created folders at: {chats_path}, {tasks_path}")
    except Exception as e:
        print(f"Failed to create folders: {str(e)}")
        raise

def save_chat_history(messages: List[Dict[str, Any]], chat_name: Optional[str] = None) -> str:
    """
    Save chat history to a JSON file.
    
    Args:
        messages: List of message dictionaries from ChatMessage.model_dump()
        chat_name: Optional name for the chat file
        
    Returns:
        str: Path to saved chat file
    """
    if not messages:
        raise ValueError("No messages to save")
        
    script_dir = Path(__file__).parent.resolve()
    chat_dir = script_dir / CHATS_FOLDER
    chat_dir.mkdir(exist_ok=True)
    
    if not chat_name:
        chat_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    file_path = chat_dir / f"{chat_name}.json"
    if file_path.exists():
        file_path = get_unique_filename(file_path)
    
    with open(file_path, 'w') as f:
        json.dump(messages, f, indent=2, default=str)
    
    logger.info(f"Saved chat history to {file_path}")
    return chat_name


def load_chat_history(chat_name: str) -> List[Dict[str, Any]]:
    """
    Load chat history from a JSON file.
    
    Args:
        chat_name: Name of the chat file to load
        
    Returns:
        List[Dict[str, Any]]: List of message dictionaries to construct ChatMessage objects
    """
    if not chat_name:
        raise ValueError("Chat name cannot be empty")
    
    script_dir = Path(__file__).parent.resolve()
    chat_dir = script_dir / CHATS_FOLDER
    file_path = chat_dir / f"{chat_name}.json"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Chat file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        messages = json.load(f)
    
    logger.info(f"Loaded chat history from {file_path}")
    return messages

def display_chat_menu() -> str:
    """Display chat action menu and get user choice."""
    print("\n=== Chat Actions ===")
    print("1. New message")
    print("2. Save chat")
    print("3. Clear chat")
    print("4. Load chat")
    print("5. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-5): ").strip()
        if choice in ['1', '2', '3', '4', '5']:
            return choice
        print("Invalid choice. Please enter a number between 1 and 5.")

class ModelParameters:
    """Class to store and validate model generation parameters."""
    def __init__(self):
        self.options = {
            "temperature": 0.7,  # Default temperature for response generation
            "top_p": 0.9,       # Default top-p value
            "top_k": 40,        # Default top-k value
            "num_predict": 2048  # Default max tokens for response
        }

    def model_dump(self) -> dict:
        """Convert parameters to dictionary."""
        return {"options": self.options}

    def model_dump_json(self, indent=None):
        """Convert parameters to JSON string."""
        import json
        return json.dumps(self.model_dump(), indent=indent)

async def chat_mode(client: ollama.AsyncClient, model: str) -> None:
    """Interactive chat mode with the selected model."""
    logger.info(f"Starting chat mode with model: {model}")
    
    # Initialize chat session
    session = []
    current_chat_name = None

    while True:
        try:
            choice = display_chat_menu()
            
            if choice == '1':  # New message
                user_input = input("\nYou: ").strip()
                if not user_input:
                    continue
                
                # Add user message to session
                session.append({"role": "user", "content": user_input})
                
                # Get model response with streaming
                print("\nAssistant: ", end='', flush=True)
                response_content = ""
                
                async for response in await client.chat(
                    model=model,
                    messages=session,
                    stream=True
                ):
                    if 'message' in response and 'content' in response['message']:
                        content = response['message']['content']
                        print(content, end='', flush=True)
                        response_content += content
                
                # Add assistant response to session
                session.append({"role": "assistant", "content": response_content})
                print("\n")
                
            elif choice == '2':  # Save chat
                chat_name = input("Enter chat name (press Enter for timestamp): ").strip()
                save_chat_history(session, chat_name)
                current_chat_name = chat_name
                
            elif choice == '3':  # Clear chat
                confirm = input("Clear current chat? This cannot be undone (y/n): ").lower()
                if confirm == 'y':
                    session = []
                    current_chat_name = None
                    print("Chat cleared.")
                    
            elif choice == '4':  # Load chat
                try:
                    script_dir = Path(__file__).parent.resolve()
                    chats = list((script_dir / CHATS_FOLDER).glob('*.json'))
                    if not chats:
                        print("No saved chats found.")
                        continue
                        
                    print("\nAvailable chats:")
                    for i, chat in enumerate(chats, 1):
                        print(f"{i}. {chat.stem}")
                        
                    choice = input("\nSelect chat number to load (0 to cancel): ").strip()
                    if choice.isdigit() and 0 < int(choice) <= len(chats):
                        session = load_chat_history(chats[int(choice)-1].stem)
                        current_chat_name = chats[int(choice)-1].stem
                        print(f"Loaded chat: {current_chat_name}")
                except Exception as e:
                    print(f"Error loading chat: {str(e)}")
                    
            elif choice == '5':  # Exit
                if session:
                    save = input("Save current chat before exiting? (y/n): ").lower()
                    if save == 'y':
                        chat_name = input("Enter chat name to save (press Enter for timestamp): ").strip()
                        save_chat_history(session, chat_name)
                logger.info("Exiting chat mode")
                break
                
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            print(f"\nError: {str(e)}")
            break

def get_available_lists() -> List[Path]:
    """Get list of available task lists."""
    try:
        script_dir = Path(__file__).parent.resolve()
        task_dir = script_dir / TASKS_FOLDER
        return sorted(task_dir.glob("*.json"))
    except Exception as e:
        logger.error(f"Failed to get task lists: {str(e)}")
        raise OSError(f"Unable to access task lists: {str(e)}")

def display_available_lists() -> Optional[Path]:
    """Display available task lists and get user selection."""
    try:
        lists = get_available_lists()
        if not lists:
            print("No task lists found")
            return None
            
        print("\nAvailable task lists:")
        for i, task_list in enumerate(lists, 1):
            print(f"{i}. {task_list.stem}")
            
        while True:
            choice = input("\nSelect task list number (0 to cancel): ").strip()
            if not choice.isdigit():
                print("Please enter a valid number")
                continue
                
            choice = int(choice)
            if choice == 0:
                return None
            if 1 <= choice <= len(lists):
                return lists[choice-1]
            print("Please enter a number within the valid range")
            
    except Exception as e:
        logger.error(f"Error displaying task lists: {str(e)}")
        print(f"Error: {str(e)}")
        return None

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
            continue

def view_current_tasks():
    """Display current tasks in the task list."""
    try:
        script_dir = Path(__file__).parent.resolve()
        task_path = script_dir / 'task_lists'
        if not task_path.exists():
            print("\nNo tasks found.")
            return
            
        print("\n=== Current Task Lists ===")
        task_files = list(task_path.glob('*.json'))
        
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

def select_model(models: List[OllamaModel], size_pref: ModelSize, embedding_only: bool) -> Optional[str]:
    """
    Select a model based on size preference and embedding requirements.
    
    Args:
        models: List of available models
        size_pref: Preferred model size category
        embedding_only: Whether to show only embedding models
        
    Returns:
        Optional[str]: Selected model name or None if no selection made
    """
    filtered = []
    for model in models:
        if embedding_only and not model.is_embedding:
            continue
        if not embedding_only and model.is_embedding:
            continue
        if size_pref is not None and model.size_category != size_pref:
            continue
        filtered.append(model)
    
    if not filtered:
        print("\nNo models found in this category.")
        return None
        
    print("\nAvailable Models:")
    for i, model in enumerate(filtered, 1):
        print(f"{i}. {model.name} ({model.size:.2f}GB) - {model.size_category.value}")
        
    while True:
        choice = input("\nSelect model number (0 to cancel): ").strip()
        if not choice or not choice.isdigit():
            return None
            
        num = int(choice)
        if num == 0:
            return None
        if 1 <= num <= len(filtered):
            return filtered[num-1].name
            
    return None

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
        # Get script directory and create necessary folders
        script_dir = get_script_dir()
        create_log_folders()
        create_chat_folders()
        
        # Validate and sanitize models path relative to script directory
        models_path = script_dir / "ollama_models_list.txt"
        print("Fetching available Ollama models...")
        models = await update_ollama_models_list(str(models_path))
        logger.info(f"Loaded {len(models)} models")
        
        if not models:
            print("\nNo models found! Please ensure:")
            print("1. Ollama is running (run 'ollama serve' in a terminal)")
            print("2. You have models installed (run 'ollama pull model_name')")
            print("3. Try running 'ollama list' to verify your installation")
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