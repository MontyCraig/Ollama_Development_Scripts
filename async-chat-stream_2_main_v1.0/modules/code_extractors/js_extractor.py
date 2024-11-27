import os
import re
import json

def extract_js_code(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(input_dir, filename), 'r') as file:
                content = file.read()

            # Extract the task number and name
            task_match = re.search(r'Task: (\d+)\. (.+)', content)
            if task_match:
                task_number = task_match.group(1)
                task_name = task_match.group(2)

                # Extract the JavaScript code
                js_code_match = re.search(r'```javascript\n(.*?)```', content, re.DOTALL)
                if js_code_match:
                    js_code = js_code_match.group(1)

                    # Create a new filename for the JavaScript file
                    js_filename = f"{task_number}_{task_name.replace(' ', '_')}.js"
                    # Remove any characters that are not suitable for filenames
                    js_filename = re.sub(r'[^\w\-_\. ]', '_', js_filename)
                    js_filepath = os.path.join(output_dir, js_filename)

                    # Write the JavaScript code to the new file
                    with open(js_filepath, 'w') as js_file:
                        js_file.write(js_code)

                    print(f"Extracted JavaScript code from {filename} to {js_filename}")

# Set the input and output directories
input_dir = "/Volumes/One Touch/Ollama Chatbot #1/chatbot_outputs/model_tests/Gemma2/txt_output"
output_dir = "/Volumes/One Touch/Ollama Chatbot #1/chatbot_outputs/model_tests/Gemma2/javascript_files"

# Run the extraction
extract_js_code(input_dir, output_dir)