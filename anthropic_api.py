#anthropic_api.py
import requests
import io
import base64
import json
import logging
import asyncio
from anthropic import AsyncAnthropic
import logging
import aiohttp

logger = logging.getLogger(__name__)

async def send_anthropic_request(api_key, model, system_message, user_message, messages, temperature, max_tokens, base64_images, tools=None, tool_choice=None):
    client = AsyncAnthropic(
        api_key=api_key,
        base_url="https://api.anthropic.com",
        default_headers={
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "prompt-caching-2024-07-31"
        }
    )
    
    anthropic_messages = prepare_anthropic_messages(user_message, messages, base64_images)
    
    data = {
        "model": model,
        "messages": anthropic_messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    if system_message:
        data["system"] = system_message
    
    if tools:
        data["tools"] = tools
    if tool_choice:
        data["tool_choice"] = tool_choice

    try:
        response = await client.messages.create(**data)
        
        if tools:
            # If tools were used, return the full response
            return response
        else:
            # If no tools were used, format the response to match the specified structure
            generated_text = response.content[0].text if response.content else ""
            return {
                "choices": [{
                    "message": {
                        "content": generated_text
                    }
                }]
            }
    except Exception as e:
        error_msg = f"Error: An exception occurred while processing the Anthropic request: {str(e)}"
        logger.error(error_msg)
        return {"choices": [{"message": {"content": error_msg}}]}

def detect_image_type(base64_string):
    """
    Detect the image type from a base64 string.
    """
    try:
        # Decode a small portion of the base64 string
        header = base64.b64decode(base64_string[:32])
        
        # Check for PNG signature
        if header.startswith(b'\x89PNG\r\n\x1a\n'):
            return 'image/png'
        # Check for JPEG signature
        elif header.startswith(b'\xff\xd8'):
            return 'image/jpeg'
        # Add more image type checks as needed
        else:
            return 'application/octet-stream'  # Default to binary data
    except:
        return 'application/octet-stream'  # If detection fails, assume binary data

def prepare_anthropic_messages(user_message, messages, base64_images=None):
    """
    Prepares messages for the Anthropic API, ensuring all images are included.
    
    Args:
        user_message (str): The user's message.
        messages (List[Dict[str, Any]]): Previous messages.
        base64_images (List[str], optional): Base64-encoded images.
    
    Returns:
        List[Dict[str, Any]]: Formatted messages.
    """
    anthropic_messages = []
    has_images = base64_images is not None and len(base64_images) > 0

    # Prepare the user message with all images
    user_content = []
    if user_message:
        user_content.append({"type": "text", "text": user_message})

    if has_images:
        for image_data in base64_images:
            media_type = detect_image_type(image_data)
            user_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data
                }
            })

    # Ensure the first message is from the user
    if not messages or messages[0]["role"] != "user":
        if user_content:
            anthropic_messages.append({"role": "user", "content": user_content})
        else:
            # If there's no user message and no images, add a dummy user message
            anthropic_messages.append({"role": "user", "content": [{"type": "text", "text": "Hello"}]})
    else:
        # Add previous messages
        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                continue  # Skip system messages as they're handled separately

            new_message = {"role": role, "content": []}

            if isinstance(content, str):
                new_message["content"].append({"type": "text", "text": content})
            elif isinstance(content, list):
                new_message["content"] = content

            if not has_images:
                if role == "assistant":
                    new_message["cache_control"] = {"type": "permanent"}
                elif role == "user":
                    new_message["cache_control"] = {"type": "ephemeral"}

            anthropic_messages.append(new_message)

        # Add the new user message with images if it's not empty
        if user_content:
            new_user_message = {"role": "user", "content": user_content}
            if not has_images:
                new_user_message["cache_control"] = {"type": "ephemeral"}
            anthropic_messages.append(new_user_message)

    return anthropic_messages