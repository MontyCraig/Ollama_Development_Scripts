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

# Load configuration
config = {}
try:
    with open('config.yaml', 'r') as config_file:
        loaded_config = yaml.safe_load(config_file)
        if isinstance(loaded_config, dict):
            config = loaded_config
        else:
            raise ValueError("Config file does not contain a valid dictionary")
except (FileNotFoundError, yaml.YAMLError, ValueError) as e:
    print(f"Error loading config file: {e}")
    print("Using default values.")

# Set default values if not present in config
default_config = {
    'pre_prompt_path': "/Volumes/One Touch/Ollama Chatbot #1/prompts/pre_prompts/test_prompt_5_pre_prompt.txt",
    'post_prompt_path': "/Volumes/One Touch/Ollama Chatbot #1/prompts/post_prompts/test_prompt_5_post_prompt.txt",
    'default_model': "llama2",
    'input_csv_path': "/Volumes/One Touch/Ollama Chatbot #1/prompts/task_lists/test_prompt_5_task_list.csv",
    'output_csv_path': "output.csv",
    'output_base_path': "output"
}

for key, value in default_config.items():
    if key not in config:
        config[key] = value

# Load pre-prompt and post-prompt
try:
    with open(config['pre_prompt_path'], 'r') as f:
        PRE_PROMPT = f.read().strip()

    with open(config['post_prompt_path'], 'r') as f:
        POST_PROMPT = f.read().strip()
except FileNotFoundError as e:
    print(f"Error loading prompt file: {e}")
    print("Using empty prompts.")
    PRE_PROMPT = ""
    POST_PROMPT = ""

# Set up logging
logging.basicConfig(filename='task_processing.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def reformat_csv(input_path, output_path):
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
                if ' - Task #' in line:
                    task = line.split(' - Task #', 1)[-1]
                    outfile.write(f"- {task}\n")
                else:
                    outfile.write(f"- {line}\n")

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
            # Print the first 3 tasks of each group (or all if less than 3)
            for task in task_list[:3]:
                print(f"    - {task}")
            if len(task_list) > 3:
                print("    ...")
        return tasks
    except FileNotFoundError:
        print(f"The CSV file was not found: {task_path}")
        return {}
    except Exception as e:
        print(f"Error reading the CSV file: {str(e)}")
        return {}

async def speak(speaker, content):
    if speaker:
        p = await asyncio.create_subprocess_exec(speaker, content)
        await p.communicate()

def create_output_folders(base_path, model_name, prompt_name):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(base_path, 'model_tests', model_name)
    prompt_path = os.path.join(model_path, prompt_name, timestamp)
    
    folders = ['txt_output', 'json_output', 'js_output', 'php_output', 'python_output', 'yaml_output', 'xml_output', 'md_output']
    
    for folder in folders:
        os.makedirs(os.path.join(prompt_path, folder), exist_ok=True)
    
    return {folder: os.path.join(prompt_path, folder) for folder in folders}

def log_metadata(file_path, start_time, end_time, token_count, files_created):
    metadata = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration": (end_time - start_time).total_seconds(),
        "token_count": token_count,
        "files_created": files_created
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
    languages = re.findall(r'(?i)\b(javascript|php|python|html|css|sql|yaml|xml|markdown)\b', pre_prompt)
    languages = list(set(languages))  # Remove duplicates
    
    for lang in languages:
        folder_name = f"{lang.lower()}_output"
        os.makedirs(os.path.join(base_path, folder_name), exist_ok=True)
    
    return languages

async def process_task(client, task, output_folders, languages, speaker, model):
    start_time = datetime.now()
    token_count = 0
    files_created = []

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
            existing_files[lang_lower] = os.listdir(output_folders[f'{lang_lower}_output'])

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

    input_path = config['input_csv_path']
    output_path = config['output_csv_path']
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

    # Create output folders
    base_path = config['output_base_path']
    output_folders = create_output_folders(base_path, args.model, "task_output")
    languages = create_language_folders(base_path, PRE_PROMPT)

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

        start_index = last_task_index + 1 if 'last_task_index' in locals() else 0

        for i in range(start_index, total_tasks):
            task = group_tasks[i]
            if task not in completed_tasks[selected_group]:
                print(f"\nProcessing task {i + 1}/{total_tasks}: {task}")
                metadata = await process_task(client, task, output_folders, languages, speaker, args.model)
                print(f"Task completed. Duration: {metadata['duration']} seconds, Tokens: {metadata['token_count']}")
                completed_tasks[selected_group].append(task)
                with open('completed_tasks.json', 'w') as f:
                    json.dump(completed_tasks, f)
                with open('last_completed_task.json', 'w') as f:
                    json.dump({'group': selected_group, 'task': task}, f)

            continue_choice = input("Continue to the next task? (y/n): ")
            if continue_choice.lower() != 'y':
                break

        selected_group = None
        print("All tasks in this group have been completed or skipped.")
        continue_choice = input("Do you want to select another group? (y/n): ")
        if continue_choice.lower() != 'y':
            break

try:
    asyncio.run(main())
except (KeyboardInterrupt, EOFError):
    print("\nExiting...")