# Setup Guide

## System Requirements

### Hardware
- CPU: 2+ cores recommended
- RAM: 8GB minimum, 16GB+ recommended
- Storage: 20GB+ free space

### Software
- Python 3.8 or higher
- Ollama installed and running
- pip package manager

## Installation Steps

1. **Clone Repository**
```bash
git clone [repository-url]
cd async-chat-stream_2_main_v1.0
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure Ollama**
- Ensure Ollama is running:
```bash
ollama serve
```
- Verify installation:
```bash
ollama list
```

5. **Directory Setup**
```bash
mkdir -p output/model_tests
mkdir -p task_lists
```

## Configuration

### Logging
- Default log file: `async_chat_stream.log`
- Log level: INFO
- Format: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`

### Model Categories
- Small: < 5GB
- Medium: 5GB - 20GB
- Large: > 20GB

### File Paths
- Models list: `/Volumes/One Touch/Ollama_Presentation/ollama_models_list.txt`
- Output directory: `./output/model_tests/`
- Task lists: `./task_lists/`

## Usage

### Starting the Application
```bash
python async-chat-stream_2_main_v1.0.py
```

### Menu Navigation
1. Use number keys (1-5) to select options
2. Follow on-screen prompts
3. Use 'exit' to quit chat sessions
4. Use 'save' to store conversations

### Task Management
1. Create task lists in CSV format
2. Place in task_lists directory
3. Use option 2 to process tasks
4. Check output directory for results

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   - Verify Ollama is running
   - Check port 11434 is available
   - Ensure correct host configuration

2. **Model Loading Issues**
   - Check available disk space
   - Verify model exists in Ollama
   - Check network connection

3. **File Permission Errors**
   - Verify write permissions in output directories
   - Check log file permissions
   - Ensure correct user permissions

### Error Logs
- Check `async_chat_stream.log` for detailed error messages
- Enable debug logging if needed

## Maintenance

### Regular Tasks
1. Update Ollama models
2. Clean old output files
3. Rotate log files
4. Check disk space

### Updates
1. Pull latest code
2. Update dependencies
3. Check for Ollama updates

## Support
[Add support contact information] 