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
from typing import Dict, List, Optional, Union, AsyncIterator, Any, Tuple
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chat_stream.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add custom exceptions
class ChatStreamError(Exception):
    """Base exception for chat stream errors."""
    pass

class TaskProcessingError(ChatStreamError):
    """Raised when task processing fails."""
    pass

class OutputError(ChatStreamError):
    """Raised when output handling fails."""
    pass

def reformat_csv(input_path: str, output_path: str) -> None:
    """
    Reformat CSV file to structured format.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        PermissionError: If writing to output path is not permitted
    """
    print(f"Reformatting CSV file: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        current_group = None
        for line in infile:
            line = line.strip()
            if not line:
                continue
            if '. ' in line and line.split('. ', 1)[0].isdigit():
                parts = line.split('. ', 1)
                if len(parts) == 2:
                    current_group = parts[1].split(' - ', 1)[0]
                    outfile.write(f"{current_group}:\n")
            elif current_group:
                if ' - ' in line:
                    task = line.split(' - ', 1)[-1]
                else:
                    task = line
                outfile.write(f"- {task}\n")

def read_tasks(task_path):
    print(f"Reading tasks from: {task_path}")
    try:
        with open(task_path, 'r', encoding='utf-8') as file:
            content = file.readlines()

        tasks = {}
        current_group = None

        for line in content:
            line = line.strip()
            if not line:
                continue

            if line.endswith(':'):
                current_group = line.rstrip(':')
                tasks[current_group] = []
            elif current_group and line.startswith('- '):
                tasks[current_group].append(line[2:])

        if not tasks:
            print("No valid task groups found in the CSV file.")
            return {}

        print("Debug: Task groups found:")
        for group, task_list in tasks.items():
            print(f"  {group}: {len(task_list)} tasks")
        return tasks
    except FileNotFoundError:
        print(f"The CSV file was not found: {task_path}")
        return {}
    except Exception as e:
        print(f"Error reading the CSV file: {str(e)}")
        return {}

# Hard-coded pre-prompt
PRE_PROMPT = """
You are a senior level programmer with a strong background in web development, and you are an expert at building web applications from scratch.
You are also an expert at building streaming radio stations from scratch.
You are an expert at javascript, php, liquidsoap, icecast, html/css, sql and postgresql.
You are also an expert at argon2 for secure password hashing.
You will be provided with a task, and you will be expected to complete only the given task at the time
Do not use too much verbosity and explanations in your responses only code. 
We are developing a comprehensive web application for a streaming radio station platform. The application will allow users to listen to multiple stations, manage the music library, create playlists, schedule shows, and provide specific features for different user roles (admins, DJs, bands/artists, advertisers).
The platform will also offer a "station rental" SaaS component, enabling users to quickly set up their own stations using customizable templates and a subscription-based model.
Technology Stack
JavaScript: Primary programming language for front-end and back-end development
PHP: Server-side scripting language for back-end logic and API development
Liquidsoap: Audio stream generation and manipulation
Icecast: Audio streaming server for delivering radio streams to listeners
HTML/CSS: Markup and styling for the application's user interface
SQL with PostgreSQL: Relational database management for storing and retrieving application data
Argon2: Secure password hashing algorithm for user authentication
Coding Practices
When developing the application, adhere to the following coding best practices:
Write clean, well-structured, and maintainable code
Provide clear and concise documentation using comments and docstrings
Use strong typing whenever possible to catch type-related errors early
Implement robust error handling and logging mechanisms
Perform regular code reviews to maintain code quality and consistency
Write unit tests to ensure code correctness and prevent regressions
Optimize code for performance, scalability, and security
Follow industry-standard coding conventions and style guides
You will be provided with a task, and you will be expected to complete only the given task at the time in context of the task above.
"""

# Hard-coded post-prompt
POST_PROMPT = """
### The AI agent should assist in designing, developing, and refining the streaming radio station platform while adhering to these technical guidelines and coding practices. 
### The agent should provide guidance, code snippets, and architectural recommendations to ensure the application is built to a high standard.
### Remember to only complete the task at hand and not to do any other tasks. 
"""

async def speak(speaker, content):
    if speaker:
        p = await asyncio.create_subprocess_exec(speaker, content)
        await p.communicate()

def create_output_folders(base_path, model_name, prompt_name):
    model_path = os.path.join(base_path, 'model_tests', model_name)
    prompt_path = os.path.join(model_path, prompt_name)
    
    folders = ['txt_output', 'json_output', 'js_output', 'php_output', 'python_output']
    
    for folder in folders:
        os.makedirs(os.path.join(prompt_path, folder), exist_ok=True)
    
    return {folder: os.path.join(prompt_path, folder) for folder in folders}

def log_metadata(file_path, start_time, end_time, token_count):
    metadata = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration": (end_time - start_time).total_seconds(),
        "token_count": token_count
    }
    return metadata

def get_unique_filename(file_path):
    base, ext = os.path.splitext(file_path)
    counter = 1
    while os.path.exists(file_path):
        file_path = f"{base}_v{counter}{ext}"
        counter += 1
    return file_path

def create_language_folders(base_path, pre_prompt):
    languages = re.findall(r'(?i)\b(javascript|php|python|html|css|sql)\b', pre_prompt)
    languages = list(set(languages))  # Remove duplicates
    
    for lang in languages:
        folder_name = f"{lang.lower()}_output"
        os.makedirs(os.path.join(base_path, folder_name), exist_ok=True)
    
    return languages

async def process_task(client: ollama.AsyncClient, task: str, output_folders: Dict[str, str], languages: List[str], speaker: Optional[str], model: str) -> Dict[str, Any]:
    logger.info(f"Starting task processing: {task}")
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
            file_paths[lang_lower] = get_unique_filename(os.path.join(output_folders[f'{lang_lower}_output'], f"{file_name}.{lang.lower}"))

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

        except ollama.Error as e:
            logger.error(f"Model error: {str(e)}", exc_info=True)
            raise TaskProcessingError(f"Model error: {str(e)}") from e
        except IOError as e:
            logger.error(f"Output handling error: {str(e)}", exc_info=True)
            raise OutputError(f"Output handling error: {str(e)}") from e

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

    logger.debug(f"Task metadata: {metadata}")

    return metadata

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--speak', default=False, action='store_true')
    parser.add_argument('--model', default='qwen2:7b', help='Model to use for chat')
    args = parser.parse_args()

    speaker = None
    if not args.speak:
        pass
    elif say := shutil.which('say'):
        speaker = say
    elif (espeak := shutil.which('espeak')) or (espeak := shutil.which('espeak-ng')):
        speaker = espeak

    client = ollama.AsyncClient()

    # Use relative paths from script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "prompts", "task_lists", "test_prompt_5_task_list.csv")
    output_path = os.path.join(script_dir, "prompts", "task_lists", "reformatted_task_list.csv")
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(input_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    reformat_csv(input_path, output_path)
    task_path = output_path
    try:
        tasks = read_tasks(task_path)
    except FileNotFoundError:
        print(f"Task file not found: {task_path}")
        print("Please make sure the file exists and the path is correct.")
        return

    if not tasks:
        print("No tasks found in the task file.")
        return

    print("Available task groups:")
    group_list = list(tasks.keys())
    for i, group in enumerate(group_list, 1):
        print(f"{i}. {group}")

    # Create output folders in script directory
    base_path = os.path.join(script_dir, "output")
    output_folders = create_output_folders(base_path, args.model, "task_output")
    languages = create_language_folders(base_path, PRE_PROMPT)

    while True:
        try:
            group_choice = input("Enter the number of the task group you want to start with (or '0' to exit): ")
            if group_choice == '0':
                print("Exiting...")
                break
            group_choice = int(group_choice) - 1
            if 0 <= group_choice < len(group_list):
                selected_group = group_list[group_choice]
                group_tasks = tasks[selected_group]
                total_tasks = len(group_tasks)
                print(f"\nSelected group: {selected_group} ({total_tasks} tasks)")

                if total_tasks == 0:
                    print("This group has no tasks. Please select another group.")
                    continue

                while True:
                    task_choice = input(f"Enter a task number (1-{total_tasks}), 'list' to see task names, 'all' to run all tasks, 'back' to select a different group, or 'exit' to quit: ")

                    if task_choice.lower() == 'list':
                        for i, task in enumerate(group_tasks, 1):
                            print(f"{i}. {task}")
                    elif task_choice.lower() == 'all':
                        for i, task in enumerate(group_tasks, 1):
                            print(f"\nProcessing task {i}/{total_tasks}: {task}")
                            metadata = await process_task(client, task, output_folders, languages, speaker, args.model)
                            print(f"Task completed. Duration: {metadata['duration']} seconds, Tokens: {metadata['token_count']}")
                        break
                    elif task_choice.lower() == 'back':
                        break
                    elif task_choice.lower() == 'exit':
                        print("Exiting...")
                        return
                    else:
                        try:
                            task_choice = int(task_choice) - 1
                            if 0 <= task_choice < total_tasks:
                                selected_task = group_tasks[task_choice]
                                print(f"Selected task: {selected_task}")
                                print(f"\nProcessing task {task_choice + 1}/{total_tasks}: {selected_task}")
                                metadata = await process_task(client, selected_task, output_folders, languages, speaker, args.model)
                                print(f"Task completed. Duration: {metadata['duration']} seconds, Tokens: {metadata['token_count']}")
                            else:
                                print("Invalid task number. Please try again.")
                        except ValueError:
                            print("Invalid input. Please enter a valid task number, 'list', 'all', 'back', or 'exit'.")
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a valid number or '0' to exit.")

try:
    asyncio.run(main())
except (KeyboardInterrupt, EOFError):
    print("\nExiting...")
