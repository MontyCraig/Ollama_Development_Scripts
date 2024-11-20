import subprocess
import json
import ollama
from datetime import datetime
import os
import nltk
from nltk.tokenize import sent_tokenize

# Download the punkt tokenizer if not already downloaded
nltk.download('punkt')

def get_ollama_list():
    """Runs the `ollama list` command and saves the output to a JSON file."""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        keys = lines[0].split()
        models = []
        for line in lines[1:]:
            values = line.split()
            model = dict(zip(keys, values))
            models.append(model)
        
        with open('ollama_models.json', 'w') as f:
            json.dump(models, f, indent=4)
        
        print("Ollama model list saved to ollama_models.json")
    except subprocess.CalledProcessError as e:
        print(f"Error running `ollama list`: {e}")

def stream_response(model_name, messages):
    """Streams the response from the model."""
    try:
        stream = ollama.chat(model=model_name, messages=messages, stream=True)
        response = ""
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
            response += chunk['message']['content']
        return response
    except Exception as e:
        print(f"Error getting streaming response from model: {e}")
        return ""

def show_model_details(model_name):
    """Shows the details of a specific model."""
    try:
        details = ollama.show(model_name)
        print(json.dumps(details, indent=4))
    except Exception as e:
        print(f"Error showing model details: {e}")

def handle_conversation():
    get_ollama_list()

    with open('ollama_models.json', 'r') as f:
        models = json.load(f)

    print("Available Ollama Models:")
    for index, model in enumerate(models):
        print(f"{index + 1}. Name: {model['NAME']}, Size: {model['SIZE']}")

    model_index = int(input("Select a model by number (or 0 to exit): ")) - 1

    if model_index == -1:
        print("Exiting the program.")
        return

    if model_index < 0 or model_index >= len(models):
        print("Invalid selection. Please select a valid model number.")
        return

    model_name = models[model_index]['NAME']

    if not model_name:
        print("Selected model name is empty. Please select a valid model.")
        return

    print(f"Welcome to the AI chat bot. I am the {model_name}. What can I help you with today? Type 'exit' to quit.")

    context = ""
    conversation_log = []
    first_prompt_subject = ""

    # Create output directories if they don't exist
    convos_dir = '/Volumes/One Touch/Ollama Chatbot #1/chatbot/Convos'
    json_outputs_dir = '/Volumes/One Touch/Ollama Chatbot #1/chatbot/JSON_Outputs'
    os.makedirs(convos_dir, exist_ok=True)
    os.makedirs(json_outputs_dir, exist_ok=True)

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting the chat.")
            break

        # Prepare the message for the chat
        messages = [{'role': 'user', 'content': user_input}]
        
        # Get the response from the model
        bot_response = stream_response(model_name, messages)
        if not bot_response:
            continue

        print(f"Bot: {bot_response}")

        # Parse the bot response into sentences
        sentences = sent_tokenize(bot_response)

        # Save conversation details
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subject = ' '.join(user_input.split()[:5])  # First five words of the first message
        if not first_prompt_subject:
            first_prompt_subject = subject  # Set the subject for the first prompt

        for sentence in sentences:
            conversation_log.append({
                'timestamp': timestamp,
                'user_message': user_input,
                'bot_response': sentence,
                'subject': subject
            })

        context += f"\nUser: {user_input}\nAI: {bot_response}"

    # Create filenames without date and time
    json_file_name = f"{model_name}_{first_prompt_subject}.json"
    text_file_name = f"{model_name}_{first_prompt_subject}.txt"

    # Save the conversation log to a JSON file
    json_file_path = os.path.join(json_outputs_dir, json_file_name)
    with open(json_file_path, 'w') as log_file:
        json.dump(conversation_log, log_file, indent=4)

    # Save the conversation in plain text format with headers
    text_file_path = os.path.join(convos_dir, text_file_name)
    with open(text_file_path, 'w') as text_file:
        text_file.write(f"Model: {model_name}\n")
        text_file.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}\n")
        text_file.write(f"Time: {datetime.now().strftime('%H:%M:%S')}\n")
        text_file.write(f"Subject: {first_prompt_subject}\n\n")
        for entry in conversation_log:
            text_file.write(f"{entry['timestamp']} - User: {entry['user_message']}\n")
            text_file.write(f"{entry['timestamp']} - Bot: {entry['bot_response']}\n\n")

if __name__ == "__main__":
    handle_conversation()