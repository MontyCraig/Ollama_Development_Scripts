"""
System-wide constants for the async chat stream system.
"""
from pathlib import Path

# Folder names
LOGS_FOLDER = "logs"
CHATS_FOLDER = "chats"
TASKS_FOLDER = "task_lists"
MODELS_FOLDER = "models"

# File extensions
CHAT_FILE_EXT = ".json"
TASK_FILE_EXT = ".csv"
LOG_FILE_EXT = ".log"

# Prompts
PRE_PROMPT = """
You will be provided with a task, and you will be expected to complete only the given task at the time in context of the task above.
"""

POST_PROMPT = """
### The AI agent should assist in designing, developing, and refining the streaming radio station platform while adhering to these technical guidelines and coding practices. 
### The agent should provide guidance, code snippets, and architectural recommendations to ensure the application is built to a high standard.
### Remember to only complete the task at hand and not to do any other tasks. 
"""

# Logging format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'

# API endpoints
OLLAMA_API_HOST = 'http://localhost:11434' 