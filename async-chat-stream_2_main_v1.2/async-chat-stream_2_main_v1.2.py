import shutil
import asyncio
import argparse
import ollama
from ollama import AsyncClient  # Add this line
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
import aiohttp

# Set up logging
logging.basicConfig(filename='task_processing.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
config = {}
try:
    with open('config.yaml', 'r') as config_file:
        loaded_config = yaml.safe_load(config_file)
        if isinstance(loaded_config, dict):
            config = loaded_config
        else:
            raise ValueError("Config file does not contain a valid dictionary")
except FileNotFoundError:
    logging.error("Config file not found. Using default values.")
    print("Error loading config file: File not found. Using default values.")
except yaml.YAMLError as e:
    logging.error(f"Error parsing YAML config file: {e}")
    print(f"Error loading config file: YAML error: {e}. Using default values.")
except ValueError as e:
    logging.error(f"Error in config file format: {e}")
    print(f"Error loading config file: {e}. Using default values.")

# Set default values if not present in config
default_config = {
    'pre_prompt_path': "/path/to/pre_prompt.txt",
    'post_prompt_path': "/path/to/post_prompt.txt",
    'default_model': "llama2",
    'input_csv_path': "/path/to/input.csv",
    'output_csv_path': "output.csv",
    'output_base_path': "output"
}

for key, value in default_config.items():
    if key not in config:
        config[key] = value

# Validate file paths
def validate_file_path(path, is_output=False):
    if not os.path.isfile(path):
        if is_output:
            # Ensure the directory for the output file exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
        else:
            raise FileNotFoundError(f"File not found: {path}")

try:
    validate_file_path(config['input_csv_path'])
    validate_file_path(config['output_csv_path'], is_output=True)
except FileNotFoundError as e:
    logging.error(str(e))
    print(str(e))
    exit(1)

# Load pre-prompt and post-prompt
PRE_PROMPT = ""
POST_PROMPT = ""
try:
    with open(config['pre_prompt_path'], 'r') as f:
        PRE_PROMPT = f.read().strip()
except FileNotFoundError:
    logging.error(f"Pre-prompt file not found: {config['pre_prompt_path']}")
    print(f"Error loading pre-prompt file: {config['pre_prompt_path']}. Using empty pre-prompt.")
except Exception as e:
    logging.error(f"Error reading pre-prompt file: {e}")
    print(f"Error loading pre-prompt file: {e}. Using empty pre-prompt.")

try:
    with open(config['post_prompt_path'], 'r') as f:
        POST_PROMPT = f.read().strip()
except FileNotFoundError:
    logging.error(f"Post-prompt file not found: {config['post_prompt_path']}")
    print(f"Error loading post-prompt file: {config['post_prompt_path']}. Using empty post-prompt.")
except Exception as e:
    logging.error(f"Error reading post-prompt file: {e}")
    print(f"Error loading post-prompt file: {e}. Using empty post-prompt.")

def reformat_csv(input_path, output_path):
    logging.debug(f"Entering function: reformat_csv")
    logging.debug(f"Reformatting CSV file: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        current_group = None
        for line in infile:
            line = line.strip()
            if not line:
                continue
            if line.startswith('@GROUP:'):
                current_group = line
                outfile.write(f"{current_group}\n")
            elif current_group and ' - Task #' in line:
                outfile.write(f"{line}\n")

def read_tasks(task_path):
    logging.debug(f"Entering function: read_tasks")
    logging.debug(f"Reading tasks from: {task_path}")
    try:
        with open(task_path, 'r', encoding='utf-8') as file:
            content = file.readlines()
        tasks = {}
        current_group = None
        for line in content:
            line = line.strip()
            if not line:
                continue
            if line.startswith('@GROUP:'):
                current_group = line[7:].strip()
                tasks[current_group] = []
            elif current_group and ' - Task #' in line:
                # Remove duplicate numbers and clean up the task description
                parts = line.split(' - Task #')
                task_number = parts[0].split('. ')[-1].strip()
                task_description = parts[1].strip()
                clean_task = f"{task_number} - Task #{task_description}"
                tasks[current_group].append(clean_task)
        if not tasks:
            logging.warning("No valid task groups found in the CSV file.")
            return {}
        logging.debug("Debug: Task groups found:")
        for group, task_list in tasks.items():
            logging.debug(f"  {group}: {len(task_list)} tasks")
        return tasks
    except FileNotFoundError:
        logging.error(f"The CSV file was not found: {task_path}")
        return {}
    except Exception as e:
        logging.error(f"Error reading the CSV file: {str(e)}", exc_info=True)
        return {}

def create_output_folders(base_path, model_name, output_type):
    folders = {
        'json_output': os.path.join(base_path, 'chatbot_outputs', 'model_tests', model_name, output_type, 'json_output'),
        'txt_output': os.path.join(base_path, 'chatbot_outputs', 'model_tests', model_name, output_type, 'txt_output'),
    }
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    return folders

def create_language_folders(base_path, model_name, output_type):
    languages = ['python', 'javascript', 'html', 'css', 'sql', 'php']  # Add more languages as needed
    folders = {}
    for lang in languages:
        folder_path = os.path.join(base_path, 'chatbot_outputs', 'model_tests', model_name, output_type, f'{lang}_output')
        os.makedirs(folder_path, exist_ok=True)
        folders[f'{lang}_output'] = folder_path
    return languages, folders

def get_unique_filename(file_path):
    base, extension = os.path.splitext(file_path)
    counter = 1
    while os.path.exists(file_path):
        file_path = f"{base}_{counter}{extension}"
        counter += 1
    return file_path

import aiohttp
import json

async def process_task(task, output_folders, languages, model):
    logging.info(f"Processing task: {task}")
    start_time = datetime.now()
    token_count = 0
    files_created = []

    # Hardcode the model name
    model = "qwen2:7b"

    # Extract task number and description
    task_parts = task.split(' - Task #')
    task_number = task_parts[0].strip()
    task_description = task_parts[1].strip()

    messages = [
        {'role': 'system', 'content': PRE_PROMPT},
        {'role': 'user', 'content': f"Task {task_number}: {task_description}"},
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
            logging.info(f"Attempting to chat with model: {model}")
            async with aiohttp.ClientSession() as session:
                async with session.post('http://localhost:11434/api/chat', json={
                    'model': model,
                    'messages': messages,
                    'stream': True
                }) as response:
                    async for line in response.content:
                        if line:
                            try:
                                chunk = json.loads(line)
                                logging.debug(f"Received chunk: {chunk}")
                                
                                if 'error' in chunk:
                                    if "not found" in chunk['error']:
                                        logging.warning(f"Model '{model}' not found. Attempting to pull...")
                                        await pull_model(session, model)
                                        logging.info(f"Model '{model}' pulled successfully. Retrying the task.")
                                        return await process_task(task, output_folders, languages, model)
                                    else:
                                        raise Exception(chunk['error'])
                                
                                if chunk.get('done'):
                                    break
                                
                                content = chunk.get('message', {}).get('content') or chunk.get('response', '')
                                
                                if not content:
                                    logging.warning(f"No content found in chunk: {chunk}")
                                    continue

                                # Check for unexpected content
                                if content.strip().startswith(('#', ')')):
                                    logging.warning(f"Unexpected content received: {content}")
                                    continue

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
                                message['content'] += content
                                token_count += 1
                            except json.JSONDecodeError:
                                logging.error(f"Failed to decode JSON: {line}")
                            except KeyError as e:
                                logging.error(f"KeyError in chunk: {e}")
                            except Exception as e:
                                logging.error(f"Unexpected error processing chunk: {str(e)}")

            logging.info("Task processing completed.")

        except aiohttp.ClientError as e:
            logging.error(f"Aiohttp ClientError: {str(e)}", exc_info=True)
            print(f"Network error: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error in process_task: {str(e)}", exc_info=True)
            print(f"An unexpected error occurred: {str(e)}")
            return None
        finally:
            for file in lang_files.values():
                file.close()

    # Parse the output and extract code snippets
    code_snippets = {}
    for lang in languages:
        lang_lower = lang.lower()
        pattern = re.compile(f'```{lang}\n(.*?)```', re.DOTALL | re.IGNORECASE)
        snippets = pattern.findall(message['content'])
        if snippets:
            code_snippets[lang.lower] = snippets

    # Get the list of existing files in the output folders
    existing_files = {}
    for lang in languages:
        lang_lower = lang.lower()
        if f'{lang.lower}_output' in output_folders:
            existing_files[lang] = os.listdir(output_folders[f'{lang.lower}_output'])

    # Write code snippets to separate files
    for lang, snippets in code_snippets.items():
        if f'{lang}_output' in output_folders:
            for i, snippet in enumerate(snippets, 1):
                file_name = f"{file_name}_{lang}_{i}.{lang}"
                if file_name not in existing_files[lang]:
                    file_path = os.path.join(output_folders[f'{lang}_output'], file_name)
                    with open(file_path, "w") as f:
                        f.write(snippet)
                    files_created.append(file_path)

    metadata = log_metadata(file_paths['txt'], start_time, datetime.now(), token_count, files_created)

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

async def pull_model(session, model):
    async with session.post('http://localhost:11434/api/pull', json={'name': model}) as response:
        async for line in response.content:
            if line:
                chunk = json.loads(line)
                print(f"Pulling model: {chunk.get('status', '')}")
        print("Model pull completed.")

async def main():
    logging.debug("Entering main function")
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

    client = AsyncClient()
    logging.debug(f"AsyncClient created with default settings")

    input_path = config['input_csv_path']
    output_path = config['output_csv_path']
    reformat_csv(input_path, output_path)
    task_path = output_path
    try:
        tasks = read_tasks(task_path)
        logging.debug(f"Debug: Tasks read: {len(tasks)} groups")
    except FileNotFoundError:
        logging.error(f"Task file not found: {task_path}")
        print("Please make sure the file exists and the path is correct.")
        return

    if not tasks:
        logging.warning("No tasks found in the task file.")
        return

    print("\nAvailable task groups:")
    group_list = list(tasks.keys())
    for i, group in enumerate(group_list, 1):
        task_count = len(tasks[group])
        group_number = group.split('.')[0].strip()
        group_name = group.split('.', 1)[1].strip()
        print(f"{group_number}. {group_name} ({task_count} tasks)")

    # Create output folders
    base_path = config['output_base_path']
    output_type = "test_prompt_5_task_list"  # You can make this configurable if needed
    output_folders = create_output_folders(base_path, args.model, output_type)
    languages, language_folders = create_language_folders(base_path, args.model, output_type)
    
    # Merge output_folders and language_folders
    output_folders.update(language_folders)

    completed_tasks = {}
    try:
        with open('completed_tasks.json', 'r') as f:
            completed_tasks = json.load(f)
    except FileNotFoundError:
        pass

    last_completed_task = {}
    try:
        with open('last_completed_task.json', 'r') as f:
            last_completed_task = json.load(f)
    except FileNotFoundError:
        pass

    if last_completed_task:
        print("Do you want to resume from the last completed task?")
        resume_choice = input("Enter 'y' to resume or any other key to start from the beginning: ")
        if resume_choice.lower() == 'y':
            selected_group = last_completed_task['group']
            last_task_index = tasks[selected_group].index(last_completed_task['task'])
            print(f"Resuming from group: {selected_group}, task: {last_completed_task['task']}")
        else:
            selected_group = None
    else:
        selected_group = None

    while True:
        if not selected_group:
            print("\nAvailable task groups:")
            for i, group in enumerate(group_list, 1):
                task_count = len(tasks[group])
                group_number = group.split('.')[0].strip()
                group_name = group.split('.', 1)[1].strip()
                print(f"{group_number}. {group_name} ({task_count} tasks)")
            try:
                group_choice = input("Enter the number of the task group you want to start with (or '0' to exit): ")
                if group_choice == '0':
                    print("Exiting...")
                    break
                group_choice = int(group_choice) - 1
                if 0 <= group_choice < len(group_list):
                    selected_group = group_list[group_choice]
                else:
                    print("Invalid choice. Please try again.")
                    continue
            except ValueError:
                print("Invalid input. Please enter a valid number or '0' to exit.")
                continue

        group_tasks = tasks[selected_group]
        total_tasks = len(group_tasks)
        print(f"\nSelected group: {selected_group} ({total_tasks} tasks)")

        if total_tasks == 0:
            print("This group has no tasks. Please select another group.")
            selected_group = None
            continue

        if selected_group not in completed_tasks:
            completed_tasks[selected_group] = []

        start_index = 0
        while start_index < total_tasks:
            print("\nTasks:")
            for i in range(start_index, min(start_index + 10, total_tasks)):
                task = group_tasks[i]
                task_number = task.split(' - Task #')[0].strip()
                status = "Completed" if task in completed_tasks[selected_group] else "Pending"
                # task_description = "No description available"  # Or extract it from the task string if available
                task_description = task.split(' - Task #')[1].strip()
                print(f"{i + 1}. Task {task_number} - {status} - {task_description}")
            
            if start_index + 10 < total_tasks:
                print("... (more tasks available)")

            task_choice = input("Enter task number to process, 'n' for next page, 'p' for previous page, 'a' to run all remaining tasks, or 'b' to go back to group selection: ")
            
            if task_choice.lower() == 'n':
                start_index = min(start_index + 10, total_tasks - 10)
                continue
            elif task_choice.lower() == 'p':
                start_index = max(0, start_index - 10)
                continue
            elif task_choice.lower() == 'b':
                break
            elif task_choice.lower() == 'a':
                # Run all remaining tasks
                for i in range(start_index, total_tasks):
                    task = group_tasks[i]
                    task_description = task.split(' - Task #')[1].strip()
                    print(f"{task_description}")
                    if task not in completed_tasks[selected_group]:
                        task_number = task.split(' - Task #')[0].strip()
                        task_description = task.split(' - Task #')[1].strip()
                        print(f"\nProcessing task {i + 1}/{total_tasks}: {task_number} ")
                        try:
                            metadata = await process_task(task, output_folders, languages, args.model)
                            if metadata:
                                print(f"Task completed. Duration: {metadata['duration']} seconds, Tokens: {metadata['token_count']}")
                                completed_tasks[selected_group].append(task)
                                with open('completed_tasks.json', 'w') as f:
                                    json.dump(completed_tasks, f)
                                with open('last_completed_task.json', 'w') as f:
                                    json.dump({'group': selected_group, 'task': task}, f)
                            else:
                                print("Task processing failed. Moving to the next task.")
                                logging.error(f"Task processing failed: {task}")
                        except Exception as e:
                            logging.error(f"Error processing task: {str(e)}", exc_info=True)
                            print(f"An error occurred while processing the task: {str(e)}")
                    else:
                        print(f"Task {i + 1} has already been completed. Skipping.")
                break  # Exit the loop after processing all tasks
            
            try:
                task_index = int(task_choice) - 1
                if 0 <= task_index < total_tasks:
                    task = group_tasks[task_index]
                    task_number = task.split(' - Task #')[0].strip()
                    task_description = "No description available"  # Or extract it from the task string if available
                    if task not in completed_tasks[selected_group]:
                        print(f"\nProcessing task {task_index + 1}/{total_tasks}: {task_number} - {task_description}")
                        try:
                            metadata = await process_task(task, output_folders, languages, args.model)
                            if metadata:
                                print(f"Task completed. Duration: {metadata['duration']} seconds, Tokens: {metadata['token_count']}")
                                completed_tasks[selected_group].append(task)
                                with open('completed_tasks.json', 'w') as f:
                                    json.dump(completed_tasks, f)
                                with open('last_completed_task.json', 'w') as f:
                                    json.dump({'group': selected_group, 'task': task}, f)
                            else:
                                print("Task processing failed.")
                                logging.error(f"Task processing failed: {task}")
                        except Exception as e:
                            logging.error(f"Error processing task: {str(e)}", exc_info=True)
                            print(f"An error occurred while processing the task: {str(e)}")
                    else:
                        print("This task has already been completed. Skipping.")
                else:
                    print("Invalid task number. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a valid task number.")

        selected_group = None
        print("All tasks in this group have been completed or skipped.")
        continue_choice = input("Do you want to select another group? (y/n): ")
        if continue_choice.lower() != 'y':
            break

try:
    asyncio.run(main())
except (KeyboardInterrupt, EOFError):
    print("\nExiting...")

