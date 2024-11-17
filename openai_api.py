#openai_api.py
import aiohttp
import json
import logging
from typing import List, Union, Optional, Dict, Any
import asyncio
import requests 
import base64
import os
logger = logging.getLogger(__name__)

async def create_openai_compatible_embedding(api_base: str, model: str, input: Union[str, List[str]], api_key: Optional[str] = None) -> List[float]:
    """
    Create embeddings using an OpenAI-compatible API asynchronously.
    
    :param api_base: The base URL for the API
    :param model: The name of the model to use for embeddings
    :param input: A string or list of strings to embed
    :param api_key: The API key (if required)
    :return: A list of embeddings
    """
    # Normalize the API base URL
    api_base = api_base.rstrip('/')
    if not api_base.endswith('/v1'):
        api_base += '/v1'
    
    url = f"{api_base}/embeddings"
    
    headers = {
        "Content-Type": "application/json"
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    payload = {
        "model": model,
        "input": input,
        "encoding_format": "float"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                
                if "data" in result and len(result["data"]) > 0 and "embedding" in result["data"][0]:
                    return result["data"][0]["embedding"] # Return the embedding directly as a list
                elif "data" in result and len(result["data"]) == 0: # handle no data in embedding result from API
                    raise ValueError("No embedding generated for the input text.")
                else:
                    raise ValueError("Unexpected response format: 'embedding' data not found")
    except aiohttp.ClientError as e:
        raise RuntimeError(f"Error calling embedding API: {str(e)}")

async def send_openai_request(api_url, base64_images, model, system_message, user_message, messages, api_key, 
                        seed, temperature, max_tokens, top_p, repeat_penalty, tools=None, tool_choice=None):
    """
    Sends a request to the OpenAI API and returns a unified response format.

    Args:
        api_url (str): The OpenAI API endpoint URL.
        base64_images (List[str]): List of images encoded in base64.
        model (str): The model to use.
        system_message (str): System message for the LLM.
        user_message (str): User message for the LLM.
        messages (List[Dict[str, Any]]): Conversation messages.
        api_key (str): API key for OpenAI.
        seed (Optional[int]): Random seed.
        temperature (float): Temperature for randomness.
        max_tokens (int): Maximum tokens to generate.
        top_p (float): Top P for sampling.
        repeat_penalty (float): Penalty for repetition.
        tools (Optional[Any], optional): Tools to be used.
        tool_choice (Optional[Any], optional): Tool choice.

    Returns:
        Union[str, Dict[str, Any]]: Standardized response.
    """
    try:
        openai_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Prepare messages
        openai_messages = prepare_openai_messages(base64_images, system_message, user_message, messages)

        data = {
            "model": model,
            "messages": openai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "presence_penalty": repeat_penalty,
            "top_p": top_p,
        }

        if seed is not None:
            data["seed"] = seed
        if tools:
            data["tools"] = tools
        if tool_choice:
            data["tool_choice"] = tool_choice


        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=openai_headers, json=data) as response:
                response.raise_for_status()
                response_data = await response.json()
                
                if tools:
                    return response_data
                else:
                    choices = response_data.get('choices', [])
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
                        error_msg = "Error: No valid choices in the OpenAI response."
                        logger.error(error_msg)
                        return {"choices": [{"message": {"content": error_msg}}]}
    except aiohttp.ClientResponseError as e:
        error_msg = f"HTTP error occurred: {e.status}, message='{e.message}', url={e.request_info.real_url}"
        logger.error(error_msg)
        return {"choices": [{"message": {"content": error_msg}}]}
    except asyncio.CancelledError:
        # Handle task cancellation if needed
        raise
    except Exception as e:
        error_msg = f"Exception during OpenAI API call: {str(e)}"
        logger.error(error_msg)
        return {"choices": [{"message": {"content": error_msg}}]}

def prepare_openai_messages(base64_images, system_message, user_message, messages):
    openai_messages = []
    
    if system_message:
        openai_messages.append({"role": "system", "content": system_message})
    
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        if role == "system":
            openai_messages.append({"role": "system", "content": content})
        elif role == "user":
            openai_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            openai_messages.append({"role": "assistant", "content": content})
    
    # Add the current user message with all images if provided
    if base64_images:
        content = [{"type": "text", "text": user_message}]
        for base64_image in base64_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                }
            })
        openai_messages.append({
            "role": "user",
            "content": content
        })
        print(f"Number of images sent: {len(base64_images)}")
    else:
        openai_messages.append({"role": "user", "content": user_message})
    
    return openai_messages

async def generate_image(
    prompt: str,
    model: str = "dall-e-3",
    n: int = 1,
    size: str = "1024x1024",
    api_key: Optional[str] = None
) -> List[str]:
    """
    Generate images from a text prompt using DALL·E.

    :param prompt: The text prompt to generate images from.
    :param model: The model to use ("dall-e-3" or "dall-e-2").
    :param n: Number of images to generate.
    :param size: Size of the generated images.
    :param api_key: The OpenAI API key.
    :return: List of image URLs or Base64 strings.
    """
    api_url = "https://api.openai.com/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "prompt": prompt,
        "n": n,
        "size": size,
        "response_format": "b64_json"
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"OpenAI generate_image error {response.status}: {error_text}")
                response.raise_for_status()
            data = await response.json()
            
            # Handle different response structures
            if "data" in data and isinstance(data["data"], list):
                images = []
                for item in data["data"]:
                    if "b64_json" in item:
                        images.append(item["b64_json"])
                    elif "url" in item:
                        # If response_format was changed or API returned URLs
                        images.append(item["url"])
                return images
            else:
                logger.error("Unexpected response format in generate_image")
                raise ValueError("Unexpected response format")

async def edit_image(
    image_base64: str,
    mask_base64: str,
    prompt: str,
    model: str = "dall-e-2",
    n: int = 1,
    size: str = "1024x1024",
    api_key: Optional[str] = None
) -> List[str]:
    """
    Edit an existing image by replacing areas defined by a mask using DALL·E.

    :param image_base64: Base64-encoded original image.
    :param mask_base64: Base64-encoded mask image.
    :param prompt: The text prompt describing the desired edits.
    :param model: The model to use ("dall-e-2").
    :param n: Number of edited images to generate.
    :param size: Size of the generated images.
    :param api_key: The OpenAI API key.
    :return: List of edited image URLs or Base64 strings.
    """
    api_url = "https://api.openai.com/v1/images/edits"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "prompt": prompt,
        "n": n,
        "size": size,
        "response_format": "b64_json"
    }
    files = {
        "image": image_base64,
        "mask": mask_base64
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, headers=headers, json=payload) as response:
            response.raise_for_status()
            data = await response.json()
            images = [item["b64_json"] for item in data.get("data", [])]
            return images

async def generate_image_variations(
    image_base64: str,
    model: str = "dall-e-2",
    n: int = 1,
    size: str = "1024x1024",
    api_key: Optional[str] = None
) -> List[str]:
    """
    Generate variations of an existing image using DALL·E.

    :param image_base64: Base64-encoded original image.
    :param model: The model to use ("dall-e-2").
    :param n: Number of variations to generate.
    :param size: Size of the generated images.
    :param api_key: The OpenAI API key.
    :return: List of variation image URLs or Base64 strings.
    """
    api_url = "https://api.openai.com/v1/images/variations"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "n": n,
        "size": size,
        "response_format": "b64_json"
    }
    files = {
        "image": image_base64
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, headers=headers, json=payload) as response:
            response.raise_for_status()
            data = await response.json()
            images = [item["b64_json"] for item in data.get("data", [])]
            return images

async def text_to_speech(text: str, model: str = "tts-1", voice: str = "alloy", response_format: str = "mp3", output_path: str = "speech.mp3", api_key: Optional[str] = None) -> None:
    """
    Convert text to spoken audio using OpenAI's TTS API.

    :param text: The text to be converted to speech.
    :param model: The TTS model to use ("tts-1" or "tts-1-hd").
    :param voice: The voice to use for audio generation.
    :param response_format: The format of the output audio ("mp3", "opus", "aac", etc.).
    :param output_path: The file path to save the generated audio.
    :param api_key: The OpenAI API key.
    """
    api_url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": response_format
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, headers=headers, json=payload) as response:
            response.raise_for_status()
            if response_format == "mp3":
                audio_data = await response.read()
                with open(output_path, "wb") as audio_file:
                    audio_file.write(audio_data)
            else:
                # Handle other formats if necessary
                pass

async def transcribe_audio(file_path: str, model: str = "whisper-1", response_format: str = "text", language: Optional[str] = None, api_key: Optional[str] = None) -> Union[str, dict]:
    """
    Transcribe audio into text using OpenAI's Whisper API.

    :param file_path: Path to the audio file to transcribe.
    :param model: The Whisper model to use ("whisper-1").
    :param response_format: The format of the transcription ("text", "verbose_json", etc.).
    :param language: (Optional) The language of the audio.
    :param api_key: The OpenAI API key.
    :return: Transcribed text or detailed JSON based on response_format.
    """
    api_url = "https://api.openai.com/v1/audio/transcriptions"
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
    Translate audio into English text using OpenAI's Whisper API.

    :param file_path: Path to the audio file to translate.
    :param model: The Whisper model to use ("whisper-1").
    :param response_format: The format of the transcription ("text", "verbose_json", etc.).
    :param api_key: The OpenAI API key.
    :return: Translated text or detailed JSON based on response_format.
    """
    api_url = "https://api.openai.com/v1/audio/translations"
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
