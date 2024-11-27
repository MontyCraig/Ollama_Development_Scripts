"""
Chat management functionality for the async chat stream system.
"""
import asyncio
import logging
from typing import List, Dict, Optional
from datetime import datetime
import ollama
from ..types.custom_types import ChatMessage, ChatSession
from ..utils.file_utils import get_unique_filename, sanitize_path
from ..config.constants import CHATS_FOLDER, CHAT_FILE_EXT

logger = logging.getLogger(__name__)

async def process_chat_message(
    client: ollama.AsyncClient,
    model: str,
    messages: List[Dict[str, str]],
    speaker: Optional[str] = None
) -> str:
    """
    Process a chat message and handle streaming response.
    
    Args:
        client: Ollama client instance
        model: Model name to use
        messages: List of chat messages
        speaker: Optional text-to-speech command
        
    Returns:
        str: Assistant's response
        
    Raises:
        ConnectionError: If chat fails
    """
    try:
        response_content = ""
        content_out = ""
        
        async for response in await client.chat(
            model=model,
            messages=messages,
            stream=True
        ):
            if response['done']:
                break
                
            content = response['message']['content']
            print(content, end='', flush=True)
            
            # Accumulate content for TTS
            content_out += content
            if content in ['.', '!', '?', '\n']:
                if speaker:
                    await speak(speaker, content_out)
                content_out = ''
                
            response_content += content
            
        if content_out and speaker:
            await speak(speaker, content_out)
            
        print()
        return response_content
        
    except Exception as e:
        logger.error(f"Chat processing error: {str(e)}")
        raise ConnectionError(f"Chat failed: {str(e)}")

async def speak(speaker: str, content: str) -> None:
    """Execute text-to-speech command."""
    if not speaker or not content:
        return
        
    try:
        p = await asyncio.create_subprocess_exec(speaker, content)
        await p.communicate()
        if p.returncode != 0:
            logger.warning(f"Speaker process failed with code {p.returncode}")
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")

def save_chat_session(session: ChatSession) -> Path:
    """
    Save chat session to file.
    
    Args:
        session: Chat session to save
        
    Returns:
        Path: Path to saved file
        
    Raises:
        IOError: If save fails
    """
    try:
        chat_dir = sanitize_path(CHATS_FOLDER)
        filename = get_unique_filename(chat_dir, CHAT_FILE_EXT, session.name)
        
        session_data = {
            "id": session.id,
            "name": session.name,
            "model_name": session.model_name,
            "messages": [msg.__dict__ for msg in session.messages],
            "created_at": session.created_at,
            "updated_at": datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
            
        logger.info(f"Saved chat session to {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Failed to save chat session: {str(e)}")
        raise IOError(f"Unable to save chat session: {str(e)}")
