import ollama
import json
from datetime import datetime
import os
import nltk
from nltk.tokenize import word_tokenize
import time
import logging
import asyncio
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# Set up logging
script_dir = Path(__file__).parent.resolve()
log_dir = script_dir / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "ollama_script.log"

logging.basicConfig(filename=str(log_file), level=logging.DEBUG,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Also log to console
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.DEBUG)
logging.getLogger('').addHandler(console)

# Download the punkt tokenizer if not already downloaded
nltk.download('punkt', quiet=True)

def sanitize_input(text: str) -> str:
    """
    Sanitize user input by removing special characters and limiting length.
    
    Args:
        text (str): Raw input text
        
    Returns:
        str: Sanitized text
    """
    # Remove special characters except basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    # Limit length to prevent injection
    return text[:1000]

def test_ollama_connection() -> bool:
    """
    Test connection to Ollama server.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        models = ollama.list()
        logging.info(f"Successfully connected to Ollama. {len(models['models'])} models available.")
        return True
    except Exception as e:
        logging.error(f"Failed to connect to Ollama: {e}")
        return False

def check_ollama_status() -> bool:
    """
    Check if Ollama service is running and responsive.
    
    Returns:
        bool: True if Ollama is running, False otherwise
    """
    try:
        ollama.list()
        logging.debug("Ollama is running and responsive")
        return True
    except Exception as e:
        logging.error(f"Ollama is not responsive: {e}")
        return False

def get_ollama_list() -> List[Dict[str, Any]]:
    """
    Get list of available Ollama models and save to JSON.
    
    Returns:
        List[Dict[str, Any]]: List of model information dictionaries
    """
    try:
        logging.debug("Fetching Ollama model list...")
        models = ollama.list()
        
        models_file = script_dir / 'ollama_models.json'
        with open(models_file, 'w') as f:
            json.dump(models, f, indent=4)
        logging.info(f"Ollama model list saved to {models_file}")
        return models['models']
    except Exception as e:
        logging.error(f"Error getting Ollama model list: {e}")
        return []

async def process_model(model_name: str, prompt: str, timeout: int = 120) -> Dict[str, Any]:
    """
    Process a single model with the given prompt.
    
    Args:
        model_name (str): Name of model to use
        prompt (str): Input prompt text
        timeout (int): Maximum time to wait for response in seconds
        
    Returns:
        Dict[str, Any]: Dictionary containing model response and metrics
    """
    logging.debug(f"Processing model: {model_name}")
    try:
        start_time = time.time()
        logging.debug(f"Sending request to model {model_name}")
        
        # Sanitize prompt
        clean_prompt = sanitize_input(prompt)
        
        response_future = asyncio.create_task(ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': clean_prompt}]
        ))
        response = await asyncio.wait_for(response_future, timeout=timeout)
        end_time = time.time()
        response_time = end_time - start_time
        
        bot_response = response['message']['content']
        token_count = len(word_tokenize(bot_response))
        
        logging.debug(f"Completed processing model: {model_name}. Response time: {response_time:.2f} seconds")
        
        return {
            'model': model_name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'prompt': clean_prompt,
            'response': bot_response,
            'response_time': response_time,
            'token_count': token_count
        }
    except asyncio.TimeoutError:
        logging.error(f"Timeout occurred while processing model {model_name}")
        return {
            'model': model_name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'prompt': clean_prompt,
            'response': "Error: Timeout occurred",
            'response_time': -1,
            'token_count': -1
        }
    except Exception as e:
        logging.error(f"Error processing model {model_name}: {e}")
        return {
            'model': model_name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'prompt': clean_prompt,
            'response': f"Error: {str(e)}",
            'response_time': -1,
            'token_count': -1
        }

def print_progress(current: int, total: int) -> None:
    """
    Print progress bar to console.
    
    Args:
        current (int): Current progress value
        total (int): Total target value
    """
    percent = (current / total) * 100
    bar_length = 50
    filled_length = int(bar_length * current // total)
    bar = '=' * filled_length + '-' * (bar_length - filled_length)
    print(f'\rProgress: [{bar}] {percent:.1f}% ({current}/{total})', end='', flush=True)

async def test_models(models: List[Dict[str, Any]], prompt: str) -> None:
    """
    Test multiple models with the same prompt.
    
    Args:
        models (List[Dict[str, Any]]): List of models to test
        prompt (str): Input prompt to test with
    """
    logging.debug(f"Starting to test {len(models)} models")
    
    # Use script directory for output
    base_dir = script_dir / 'model_tests'
    subject = ' '.join(prompt.split()[:5])  # First five words of the prompt
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{subject}_{timestamp}"
    test_dir = base_dir / folder_name
    json_dir = test_dir / 'json'
    txt_dir = test_dir / 'txt'
    
    test_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(exist_ok=True)
    txt_dir.mkdir(exist_ok=True)

    for i, model in enumerate(models, 1):
        print_progress(i, len(models))
        logging.debug(f"Processing model {i}/{len(models)}: {model['name']}")
        result = await process_model(model['name'], prompt)
        
        file_name = f"{model['name']}_{subject}"
        json_file_path = json_dir / f"{file_name}.json"
        txt_file_path = txt_dir / f"{file_name}.txt"

        with open(json_file_path, 'w') as log_file:
            json.dump(result, log_file, indent=4)

        with open(txt_file_path, 'w') as text_file:
            text_file.write(f"Model: {model['name']}\n")
            text_file.write(f"Date: {result['timestamp']}\n")
            text_file.write(f"Response Time: {result['response_time']:.2f} seconds\n")
            text_file.write(f"Token Count: {result['token_count']}\n\n")
            text_file.write(f"Prompt: {result['prompt']}\n\n")
            text_file.write(f"Response:\n{result['response']}\n")

        logging.debug(f"Completed processing model {i}/{len(models)}: {model['name']}")
        await asyncio.sleep(5)  # 5-second delay between models

    print()  # New line after progress bar
    logging.debug("Finished testing all models")

def get_prompt() -> Optional[str]:
    """
    Get multi-line prompt from user input.
    
    Returns:
        Optional[str]: Sanitized prompt text or None if input invalid
    """
    logging.debug("Waiting for user input...")
    print("Enter the prompt to test the models.")
    print("You can enter multiple lines. Type 'END' on a new line when finished:")
    prompt_lines = []
    while True:
        try:
            line = input()
            if line.strip().upper() == 'END':
                break
            prompt_lines.append(line)
            logging.debug(f"Received input line (length: {len(line)})")
        except EOFError:
            logging.warning("EOFError encountered while reading input")
            break
        except KeyboardInterrupt:
            logging.warning("KeyboardInterrupt encountered while reading input")
            return None
    
    prompt = '\n'.join(prompt_lines)
    if not prompt:
        logging.warning("Empty prompt received")
        return None
    
    # Sanitize prompt
    clean_prompt = sanitize_input(prompt)
    
    logging.debug(f"Complete prompt received. Total length: {len(clean_prompt)} characters")
    logging.debug(f"Prompt preview: {clean_prompt[:100]}...")
    return clean_prompt

async def main() -> None:
    """Main program entry point."""
    if not test_ollama_connection():
        print("Exiting due to Ollama connection failure.")
        return
    try:
        logging.debug("Script started")
        if not check_ollama_status():
            logging.error("Ollama is not running or not responsive. Please start Ollama and try again.")
            return
        models = get_ollama_list()
        logging.debug(f"Retrieved {len(models)} models")
        if not models:
            logging.error("No models available to test.")
            return

        prompt = get_prompt()
        if prompt is None:
            logging.error("No valid prompt received. Exiting.")
            return
        
        print(f"Received prompt (length: {len(prompt)} characters).")
        print(f"Preview: {prompt[:100]}...")
        confirm = input("Is this correct? (yes/no): ").strip().lower()
        if confirm != 'yes':
            logging.error("Prompt not confirmed. Exiting.")
            return
        
        logging.debug("Starting model testing...")
        await test_models(models, prompt)
        logging.debug("Testing completed.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())