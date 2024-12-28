# groq_api.py
import os
import json
import logging
from typing import List, Union, Optional, Dict, Any
import asyncio
import base64
import aiohttp
from groq import AsyncGroq, GroqError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Initialize the AsyncGroq client with your API key


async def send_groq_request(
    base64_images: List[str],
    model: str,
    system_message: str,
    user_message: str,
    messages: List[Dict[str, Any]],
    api_key: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    tools: Optional[Any] = None,
    tool_choice: Optional[Any] = None
) -> Union[str, Dict[str, Any]]:
    """
    Sends a request to the Groq API and returns a unified response format.

    Args:
        base64_images (List[str]): List of images encoded in base64.
        model (str): The model to use.
        system_message (str): System message for the LLM.
        user_message (str): User message for the LLM.
        messages (List[Dict[str, Any]]): Conversation messages.
        temperature (float): Temperature for randomness.
        max_tokens (int): Maximum tokens to generate.
        top_p (float): Top P for sampling.
        tools (Optional[Any], optional): Tools to be used.
        tool_choice (Optional[Any], optional): Tool choice.

    Returns:
        Union[str, Dict[str, Any]]: Standardized response.
    """
    try:
        client = AsyncGroq(api_key=api_key)
        # Prepare messages
        groq_messages = prepare_groq_messages(base64_images, user_message, messages)

        # Create completion using AsyncGroq client
        completion = await client.chat.completions.create(
            model=model,
            messages=groq_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=False,  # Assuming streaming is not required
            stop=None,      # Adjust stop sequences if necessary
        )

        # Convert completion to a serializable format
        completion_dict = completion.to_dict() if hasattr(completion, 'to_dict') else {}
        logger.debug(f"Received response: {json.dumps(completion_dict, indent=2)}")
        try:
            if tools:
                return completion_dict if completion_dict else completion
            else:
                choices = completion_dict.get('choices', []) if completion_dict else []
                if choices:
                    choice = choices[0]
                    message = choice.get('message', {})
                    generated_text = message.get('content', '')
                    return {
                        "choices": [{
                            "message": {
                                "content": generated_text
                            }
                        }]
                    }
                else:
                    error_msg = "Error: No valid choices in the Groq response."
                    logger.error(error_msg)
                    return {"choices": [{"message": {"content": error_msg}}]}
        except GroqError as e:
            logger.error(f"Groq API error: {e}")
            return {"choices": [{"message": {"content": str(e)}}]}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"choices": [{"message": {"content": "An unexpected error occurred."}}]}

def prepare_groq_messages(
    base64_images: List[str],
    system_message: str = "",
    user_message: str = "",
    messages: List[Dict[str, Any]] = []
) -> List[Dict[str, Any]]:
    """
    Prepares the messages in the required format for Groq API.

    Args:
        base64_images (List[str]): List of images encoded in base64.
        system_message (str, optional): System message for the LLM. Defaults to "".
        user_message (str, optional): User message for the LLM. Defaults to "".
        messages (List[Dict[str, Any]], optional): Conversation messages. Defaults to [].

    Returns:
        List[Dict[str, Any]]: Formatted messages for Groq API.
    """
    groq_messages = []
    
    # Omit system messages when images are being sent
    if not base64_images and system_message:
        groq_messages.append({"role": "system", "content": system_message})
    
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        
        if role in ["system", "user", "assistant"]:
            groq_messages.append({"role": role, "content": content})
    
    # Add the current user message with all images if provided
    if base64_images:
        content = [
            {"type": "text", "text": user_message}
        ]
        for base64_image in base64_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                }
            })
        groq_messages.append({
            "role": "user",
            "content": content
        })
        logger.debug(f"Number of images sent: {len(base64_images)}")
        for idx, base64_image in enumerate(base64_images):
            logger.debug(f"Image {idx+1} Base64 Length: {len(base64_image)}")
    else:
        if user_message:
            groq_messages.append({"role": "user", "content": user_message})
    
    return groq_messages

async def transcribe_audio(file_path: str, model: str = "whisper-1", response_format: str = "text", language: Optional[str] = None, api_key: Optional[str] = None) -> Union[str, dict]:
    """
    Transcribe audio into text using Groq's Whisper API.

    :param file_path: Path to the audio file to transcribe.
    :param model: The Whisper model to use ("whisper-1").
    :param response_format: The format of the transcription ("text", "verbose_json", etc.).
    :param language: (Optional) The language of the audio.
    :param api_key: The Groq API key.
    :return: Transcribed text or detailed JSON based on response_format.
    """
    api_url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    with open(file_path, "rb") as audio_file:
        files = {
            "file": (os.path.basename(file_path), audio_file, "audio/mpeg"),
            "model": (None, model),
            "response_format": (None, response_format)
        }
        if language:
            files["language"] = (None, language)

        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, data=files) as response:
                response.raise_for_status()
                if response_format == "text":
                    data = await response.text()
                else:
                    data = await response.json()
                return data

async def translate_audio(file_path: str, model: str = "whisper-1", response_format: str = "text", api_key: Optional[str] = None) -> Union[str, dict]:
    """
    Translate audio into English text using Groq's Whisper API.

    :param file_path: Path to the audio file to translate.
    :param model: The Whisper model to use ("whisper-1").
    :param response_format: The format of the transcription ("text", "verbose_json", etc.).
    :param api_key: The Groq API key.
    :return: Translated text or detailed JSON based on response_format.
    """
    api_url = "https://api.groq.com/openai/v1/audio/translations"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    with open(file_path, "rb") as audio_file:
        files = {
            "file": (os.path.basename(file_path), audio_file, "audio/mpeg"),
            "model": (None, model),
            "response_format": (None, response_format)
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, data=files) as response:
                response.raise_for_status()
                if response_format == "text":
                    data = await response.text()
                else:
                    data = await response.json()
                return data
            