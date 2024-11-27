from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import shutil
import asyncio
import argparse
import ollama
import os
import time
import json
from datetime import datetime
import csv
import re
import pandas as pd
from tqdm import tqdm
import yaml
import logging
from collections import defaultdict
from src.utils.path_utils import sanitize_path, get_project_root, create_directory
from src.utils.config_utils import resolve_config_paths

def setup_directories() -> Dict[str, Path]:
    """
    Initialize required directories relative to script location.
    
    Returns:
        Dict mapping directory names to Path objects
    """
    script_dir = Path(__file__).parent.absolute()
    
    # Create basic directory structure relative to script
    directories = {
        'output': script_dir / 'output',
        'config': script_dir / 'config', 
        'logs': script_dir / 'logs',
        'model_tests': script_dir / 'output/model_tests'
    }
    
    for dir_path in directories.values():
        create_directory(dir_path)
    
    return directories

def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load configuration with paths relative to script directory."""
    script_dir = Path(__file__).parent.absolute()
    
    if config_path is None:
        config_path = script_dir / 'config/config.yaml'
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        default_config = {
            'pre_prompt_path': 'config/pre_prompt.txt',
            'post_prompt_path': 'config/post_prompt.txt', 
            'default_model': 'llama2',
            'input_csv_path': 'input/task_list.csv',
            'output_csv_path': 'output/output.csv',
            'output_base_path': 'output',
            'log_level': 'INFO',
            'log_file': 'logs/async_chat_stream.log'
        }
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f)
            
        config = default_config
    else:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    
    # Resolve paths relative to script directory
    for key in config:
        if key.endswith('_path'):
            config[key] = str(script_dir / config[key])
    
    return config

# Initialize directories and logging
directories = setup_directories()
config = load_config()

# Set up logging relative to script
log_path = Path(__file__).parent.absolute() / config['log_file']
create_directory(log_path.parent)

logging.basicConfig(
    filename=str(log_path),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_prompts(pre_path: Union[str, Path], post_path: Union[str, Path]) -> tuple[str, str]:
    """
    Load pre and post prompts from files with validation.
    
    Args:
        pre_path: Path to pre-prompt file
        post_path: Path to post-prompt file
        
    Returns:
        Tuple of (pre_prompt, post_prompt) strings
        
    Raises:
        FileNotFoundError: If prompt files don't exist
    """
    logger.info("Loading prompt files")
    
    try:
        # Convert paths relative to script directory
        script_dir = Path(__file__).parent.absolute()
        pre_path = script_dir / pre_path
        post_path = script_dir / post_path
        
        with open(pre_path, 'r') as f:
            pre_prompt = f.read().strip()
        with open(post_path, 'r') as f:
            post_prompt = f.read().strip()
            
        if not pre_prompt or not post_prompt:
            raise ValueError("Prompt files cannot be empty")
            
        logger.info("Successfully loaded prompt files")
        return pre_prompt, post_prompt
        
    except FileNotFoundError as e:
        logger.error(f"Prompt file not found: {e}")
        return "", ""
    except Exception as e:
        logger.error(f"Error loading prompts: {e}")
        return "", ""

# Load prompts relative to script directory
script_dir = Path(__file__).parent.absolute()
PRE_PROMPT, POST_PROMPT = load_prompts(script_dir / config['pre_prompt_path'], 
                                     script_dir / config['post_prompt_path'])

def sanitize_text(text: str) -> str:
    """
    Sanitize text input by removing special characters and normalizing whitespace.
    
    Args:
        text: Input text to sanitize
        
    Returns:
        Sanitized text string
    """
    # Remove special characters except basic punctuation
    text = re.sub(r'[^\w\s\-.,;:!?]', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text

def reformat_csv(input_path: Union[str, Path], output_path: Union[str, Path]) -> None:
    """
    Reformat CSV file into grouped task format with validation and error handling.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to write reformatted output
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        PermissionError: If cannot write to output
    """
    logger.info(f"Reformatting CSV from {input_path} to {output_path}")
    
    # Convert paths relative to script directory
    script_dir = Path(__file__).parent.absolute()
    input_path = script_dir / input_path
    output_path = script_dir / output_path
    
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            current_group: Optional[str] = None
            
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
                            current_group = sanitize_text(current_group)
                            outfile.write(f"{current_group}:\n")
                    elif current_group:
                        if ' - Task #' in line:
                            task = line.split(' - Task #', 1)[-1]
                        else:
                            task = line
                        # Sanitize task
                        task = sanitize_text(task)
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
        Dict mapping task groups to lists of tasks
        
    Raises:
        FileNotFoundError: If task file doesn't exist
        ValueError: If file format is invalid
    """
    logger.info(f"Reading tasks from: {task_path}")
    
    # Convert path relative to script directory
    script_dir = Path(__file__).parent.absolute()
    task_path = script_dir / task_path
    
    try:
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
                        current_group = sanitize_text(current_group)
                        tasks[current_group] = []
                    elif current_group and line.startswith('- '):
                        task = line[2:]
                        # Sanitize task
                        task = sanitize_text(task)
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

async def speak(speaker: Optional[str], content: str) -> None:
    """
    Speak content using text-to-speech if speaker is provided.
    
    Args:
        speaker: Path to TTS binary
        content: Text content to speak
        
    Raises:
        RuntimeError: If speaker process fails
    """
    if not speaker:
        return
        
    try:
        logger.debug(f"Speaking content with length {len(content)}")
        # Convert speaker path relative to script directory
        script_dir = Path(__file__).parent.absolute()
        speaker_path = script_dir / speaker
        p = await asyncio.create_subprocess_exec(str(speaker_path), sanitize_text(content))
        await p.communicate()
        
    except Exception as e:
        logger.error(f"Failed to speak content: {e}")
        raise RuntimeError(f"Speaker process failed: {e}")

def create_output_folders(base_path: Union[str, Path], model_name: str, prompt_name: str) -> Dict[str, str]:
    """
    Create folder structure for model outputs with validation and sanitization.
    
    Args:
        base_path: Base directory path
        model_name: Name of the model being used
        prompt_name: Name of the prompt being processed
        
    Returns:
        Dict mapping folder types to their full paths
        
    Raises:
        ValueError: If input parameters are invalid
        OSError: If folder creation fails
    """
    logger.info(f"Creating output folders for model {model_name} and prompt {prompt_name}")
    
    # Input validation
    if not base_path or not model_name or not prompt_name:
        raise ValueError("All path components must be non-empty")
        
    # Sanitize inputs
    model_name = re.sub(r'[^\w\-\.]', '_', model_name)
    prompt_name = re.sub(r'[^\w\-\.]', '_', prompt_name)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        # Convert base path relative to script directory
        script_dir = Path(__file__).parent.absolute()
        base_path = script_dir / base_path
        model_path = base_path / 'model_tests' / model_name
        prompt_path = model_path / prompt_name / timestamp
        
        folders = ['txt_output', 'json_output', 'js_output', 'php_output', 
                  'python_output', 'yaml_output', 'xml_output', 'md_output']
        
        folder_paths = {}
        for folder in folders:
            folder_path = prompt_path / folder
            folder_path.mkdir(parents=True, exist_ok=True)
            folder_paths[folder] = str(folder_path)
            
        logger.info(f"Created {len(folders)} output folders under {prompt_path}")
        return folder_paths
        
    except OSError as e:
        logger.error(f"Failed to create output folders: {str(e)}")
        raise

def log_metadata(file_path: Union[str, Path], 
                start_time: datetime,
                end_time: datetime, 
                token_count: int,
                files_created: List[str]) -> Dict[str, Any]:
    """
    Generate metadata for task execution with validation.
    
    Args:
        file_path: Path to output file
        start_time: Task start timestamp
        end_time: Task end timestamp
        token_count: Number of tokens processed
        files_created: List of created file paths
        
    Returns:
        Dict containing metadata
        
    Raises:
        ValueError: If input parameters are invalid
    """
    logger.info(f"Generating metadata for {file_path}")
    
    # Input validation
    if not isinstance(start_time, datetime) or not isinstance(end_time, datetime):
        raise ValueError("Invalid timestamp format")
    if token_count < 0:
        raise ValueError("Token count must be non-negative")
        
    # Convert paths to be relative to script directory
    script_dir = Path(__file__).parent.absolute()
    rel_files_created = [str(Path(f).relative_to(script_dir)) for f in files_created]
        
    metadata = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration": (end_time - start_time).total_seconds(),
        "token_count": token_count,
        "files_created": rel_files_created
    }
    
    logger.debug(f"Generated metadata: {metadata}")
    return metadata

def get_unique_filename(file_path: Union[str, Path]) -> str:
    """
    Generate unique filename by appending version number.
    
    Args:
        file_path: Original file path
        
    Returns:
        Unique file path string
        
    Raises:
        ValueError: If file path is invalid
    """
    if not file_path:
        raise ValueError("File path cannot be empty")
        
    # Convert to absolute path relative to script directory
    script_dir = Path(__file__).parent.absolute()
    file_path = script_dir / Path(file_path)
    
    base = file_path.stem
    ext = file_path.suffix
    
    counter = 1
    new_path = file_path
    while new_path.exists():
        new_path = file_path.parent / f"{base}_v{counter}{ext}"
        counter += 1
        
    logger.debug(f"Generated unique filename: {new_path}")
    return str(new_path)

def create_language_folders(base_path: Union[str, Path], pre_prompt: str) -> List[str]:
    """
    Create output folders for detected programming languages.
    
    Args:
        base_path: Base directory path
        pre_prompt: Prompt text to scan for languages
        
    Returns:
        List of detected language names
        
    Raises:
        ValueError: If inputs are invalid
        OSError: If folder creation fails
    """
    logger.info("Detecting languages and creating folders")
    
    if not base_path or not pre_prompt:
        raise ValueError("Base path and pre-prompt must be non-empty")
        
    try:
        # Convert base_path to absolute path relative to script directory
        script_dir = Path(__file__).parent.absolute()
        base_path = script_dir / Path(base_path)
        
        languages = re.findall(r'(?i)\b(javascript|php|python|html|css|sql|yaml|xml|markdown)\b', 
                             pre_prompt)
        languages = list(set(lang.lower() for lang in languages))
        
        for lang in languages:
            folder_path = base_path / f"{lang}_output"
            folder_path.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Created folders for languages: {languages}")
        return languages
        
    except OSError as e:
        logger.error(f"Failed to create language folders: {str(e)}")
        raise

async def process_task(client: Any,
                      task: str, 
                      output_folders: Dict[str, str],
                      languages: List[str],
                      speaker: Optional[str],
                      model: str) -> Dict[str, Any]:
    """
    Process a single task with the language model.
    
    Args:
        client: Ollama client instance
        task: Task description
        output_folders: Dict mapping folder types to paths
        languages: List of programming languages to detect
        speaker: Optional TTS speaker path
        model: Model name to use
        
    Returns:
        Dict containing task metadata
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If task processing fails
    """
    logger.info(f"Processing task with model {model}")
    
    # Input validation
    if not task or not output_folders or not languages or not model:
        raise ValueError("Required parameters missing")
        
    start_time = datetime.now()
    token_count = 0
    files_created: List[str] = []

    messages = [
        {'role': 'system', 'content': PRE_PROMPT},
        {'role': 'user', 'content': f"Task: {task}"}, 
        {'role': 'system', 'content': POST_PROMPT},
    ]
    
    content_out = ''
    message = {'role': 'assistant', 'content': ''}

    # Sanitize filename components
    safe_task = re.sub(r'[^\w\s-]', '_', " ".join(task.split()[:5]))
    file_name = f"{safe_task}_{start_time.strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Convert all paths to be relative to script directory
        script_dir = Path(__file__).parent.absolute()
        file_paths = {
            'json': get_unique_filename(script_dir / Path(output_folders['json_output']) / f"{file_name}.json"),
            'txt': get_unique_filename(script_dir / Path(output_folders['txt_output']) / f"{file_name}.txt"),
        }
        
        for lang in languages:
            lang_lower = lang.lower()
            if f'{lang_lower}_output' in output_folders:
                file_paths[lang_lower] = get_unique_filename(
                    script_dir / Path(output_folders[f'{lang_lower}_output']) / f"{file_name}.{lang_lower}"
                )

        with open(file_paths['txt'], "w") as txt_file, \
             open(file_paths['json'], "w") as json_file:
            
            lang_files = {
                lang.lower(): open(file_paths[lang.lower()], "w") 
                for lang in languages 
                if f'{lang.lower()}_output' in output_folders
            }
            
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
                                files_created.append(file_paths[lang_lower])

                    content_out += content
                    if content in ['.', '!', '?', '\n']:
                        await speak(speaker, content_out)
                        content_out = ''

                    message['content'] += content
                    token_count += 1

                if content_out:
                    await speak(content_out)
                print()

            finally:
                for file in lang_files.values():
                    file.close()

            end_time = datetime.now()
            
            try:
                # Parse the output and extract code snippets
                code_snippets = {}
                for lang in languages:
                    lang_lower = lang.lower()
                    pattern = re.compile(f'```{lang}\n(.*?)```', re.DOTALL | re.IGNORECASE)
                    snippets = pattern.findall(message['content'])
                    if snippets:
                        code_snippets[lang_lower] = snippets

                # Get the list of existing files in the output folders
                existing_files = {}
                for lang in languages:
                    lang_lower = lang.lower()
                    if f'{lang_lower}_output' in output_folders:
                        folder_path = script_dir / Path(output_folders[f'{lang_lower}_output'])
                        existing_files[lang_lower] = os.listdir(folder_path)
            except Exception as e:
                logger.error(f"Error parsing code snippets: {e}")
                raise

    except Exception as e:
        logger.error(f"Error in process_task: {e}")
        raise

    # Write code snippets to separate files
    for lang, snippets in code_snippets.items():
        if f'{lang}_output' in output_folders:
            for i, snippet in enumerate(snippets, 1):
                file_name = f"{file_name}_{lang}_{i}.{lang}"
                if file_name not in existing_files[lang]:
                    file_path = script_dir / Path(output_folders[f'{lang}_output']) / file_name
                    with open(file_path, "w") as f:
                        f.write(snippet)
                    files_created.append(str(file_path))

    metadata = log_metadata(file_paths['txt'], start_time, end_time, token_count, files_created)
    
    with open(file_paths['txt'], "r+") as f:
        content = f.read()
        f.seek(0, 0)
        f.write(f"Task: {task}\n\nMetadata: {json.dumps(metadata, indent=2)}\n\nFiles Created:\n")
        for file in files_created:
            f.write(f"- {file}\n")
        f.write(f"\n{content}")

    try:
        with open(file_paths['json'], "w") as f:
            json.dump({"task": task, "metadata": metadata, "content": message['content']}, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving JSON file: {e}")

    return metadata

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--speak', default=False, action='store_true')
    parser.add_argument('--model', default=config['default_model'], help='Model to use for chat')
    args = parser.parse_args()

    speaker = None
    if not args.speak:
        pass
    elif say := shutil.which('say'):
        speaker = say
    elif (espeak := shutil.which('espeak')) or (espeak := shutil.which('espeak-ng')):
        speaker = espeak

    client = ollama.AsyncClient()

    # Convert paths to be relative to script directory
    script_dir = Path(__file__).parent.absolute()
    input_path = script_dir / config['input_csv_path']
    output_path = script_dir / config['output_csv_path']
    reformat_csv(input_path, output_path)
    task_path = output_path
    try:
        tasks = read_tasks(task_path)
    except FileNotFoundError:
        logging.error(f"Task file not found: {task_path}")
        print("Please make sure the file exists and the path is correct.")
        return

    if not tasks:
        logging.warning("No tasks found in the task file.")
        return

    print("Available task groups:")
    group_list = list(tasks.keys())
    for i, group in enumerate(group_list, 1):
        print(f"{i}. {group}")
    # Create output folders with validation
    try:
        base_path = script_dir / sanitize_path(config['output_base_path'])
        model_name = sanitize_model_name(args.model)
        output_folders = create_output_folders(base_path, model_name, "task_output")
        languages = create_language_folders(base_path, PRE_PROMPT)
        logger.info(f"Created output folders at {base_path} for model {model_name}")
    except (ValueError, OSError) as e:
        logger.error(f"Failed to create output folders: {e}")
        print("Error creating output folders. Please check paths and permissions.")
        return

    # Load completed tasks with validation
    completed_tasks: Dict[str, List[str]] = {}
    try:
        completed_tasks_path = script_dir / 'completed_tasks.json'
        with open(completed_tasks_path, 'r') as f:
            completed_tasks = json.load(f)
            if not isinstance(completed_tasks, dict):
                raise ValueError("Invalid completed tasks format")
            logger.info(f"Loaded {len(completed_tasks)} completed task groups")
    except FileNotFoundError:
        logger.info("No completed tasks file found - starting fresh")
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing completed tasks file: {e}")
        print("Error loading completed tasks. Starting fresh.")
    except Exception as e:
        logger.error(f"Unexpected error loading completed tasks: {e}")
        print("Error loading completed tasks. Starting fresh.")

    # Load last completed task with validation
    last_completed_task: Dict[str, str] = {}
    try:
        last_task_path = script_dir / 'last_completed_task.json'
        with open(last_task_path, 'r') as f:
            last_completed_task = json.load(f)
            if not isinstance(last_completed_task, dict) or \
               'group' not in last_completed_task or \
               'task' not in last_completed_task:
                raise ValueError("Invalid last completed task format")
            logger.info(f"Loaded last completed task: {last_completed_task}")
    except FileNotFoundError:
        logger.info("No last completed task file found")
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing last completed task file: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading last completed task: {e}")

    # Handle task resumption
    selected_group: Optional[str] = None
    if last_completed_task:
        print("Do you want to resume from the last completed task?")
        resume_choice = input("Enter 'y' to resume or any other key to start from beginning: ").strip().lower()
        if resume_choice == 'y':
            selected_group = last_completed_task['group']
            try:
                last_task_index = tasks[selected_group].index(last_completed_task['task'])
                logger.info(f"Resuming from group: {selected_group}, task: {last_completed_task['task']}")
                print(f"Resuming from group: {selected_group}, task: {last_completed_task['task']}")
            except (KeyError, ValueError) as e:
                logger.error(f"Error finding last task position: {e}")
                selected_group = None
                last_task_index = -1
        else:
            selected_group = None

    # Main task processing loop
    while True:
        if not selected_group:
            try:
                print("\nAvailable task groups:")
                for i, group in enumerate(group_list, 1):
                    print(f"{i}. {group}")
                    
                group_choice = input("\nEnter the number of the task group (0 to exit): ").strip()
                if group_choice == '0':
                    logger.info("User chose to exit")
                    print("Exiting...")
                    break
                    
                group_choice = int(group_choice) - 1
                if 0 <= group_choice < len(group_list):
                    selected_group = group_list[group_choice]
                    logger.info(f"Selected task group: {selected_group}")
                else:
                    logger.warning(f"Invalid group choice: {group_choice + 1}")
                    print("Invalid choice. Please try again.")
                    continue
            except ValueError:
                logger.warning("Invalid numeric input for group selection")
                print("Invalid input. Please enter a valid number or '0' to exit.")
                continue

        # Validate selected group tasks
        group_tasks = tasks[selected_group]
        total_tasks = len(group_tasks)
        logger.info(f"Processing group {selected_group} with {total_tasks} tasks")
        print(f"\nSelected group: {selected_group} ({total_tasks} tasks)")

        if total_tasks == 0:
            logger.warning(f"Empty task group: {selected_group}")
            print("This group has no tasks. Please select another group.")
            selected_group = None
            continue

        # Initialize completed tasks tracking for group
        if selected_group not in completed_tasks:
            completed_tasks[selected_group] = []

        # Process tasks
        start_index = last_task_index + 1 if 'last_task_index' in locals() else 0
        for i in range(start_index, total_tasks):
            task = group_tasks[i]
            if task not in completed_tasks[selected_group]:
                try:
                    print(f"\nProcessing task {i + 1}/{total_tasks}: {task}")
                    metadata = await process_task(client, task, output_folders, languages, speaker, args.model)
                    logger.info(f"Completed task: {task}, Duration: {metadata['duration']}s, Tokens: {metadata['token_count']}")
                    print(f"Task completed. Duration: {metadata['duration']} seconds, Tokens: {metadata['token_count']}")
                    
                    # Update completion tracking
                    completed_tasks[selected_group].append(task)
                    try:
                        with open(script_dir / 'completed_tasks.json', 'w') as f:
                            json.dump(completed_tasks, f)
                        with open(script_dir / 'last_completed_task.json', 'w') as f:
                            json.dump({'group': selected_group, 'task': task}, f)
                    except IOError as e:
                        logger.error(f"Failed to save completion status: {e}")
                except Exception as e:
                    logger.error(f"Error processing task: {e}")
                    print(f"Error processing task: {e}")

            continue_choice = input("Continue to next task? (y/n): ").strip().lower()
            if continue_choice != 'y':
                break

        selected_group = None
        logger.info("Completed processing current task group")
        print("All tasks in this group have been completed or skipped.")
        
        continue_choice = input("Select another group? (y/n): ").strip().lower()
        if continue_choice != 'y':
            break

try:
    asyncio.run(main())
except (KeyboardInterrupt, EOFError):
    logger.info("Application terminated by user")
    print("\nExiting...")
except Exception as e:
    logger.error(f"Application error: {e}")
    print(f"\nError: {e}")