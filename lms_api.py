#lms_api.py
import requests
import json
from typing import List, Union, Optional
import aiohttp
import asyncio
import logging
logger = logging.getLogger(__name__)

def create_lmstudio_compatible_embedding(api_base: str, model: str, input: Union[str, List[str]], api_key: Optional[str] = None) -> List[float]:
    """
    Create embeddings using an lmstudio-compatible API.
    
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
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if "data" in result and len(result["data"]) > 0 and "embedding" in result["data"][0]:
            # If multiple embeddings are returned, we'll just use the first one
            return result["data"][0]["embedding"]
        else:
            raise ValueError("Unexpected response format: 'embedding' data not found")
    except requests.RequestException as e:
        raise RuntimeError(f"Error calling embedding API: {str(e)}")
    
async def send_lmstudio_request(api_url, base64_images, model, system_message, user_message, messages, seed, temperature, 
                                max_tokens, top_k, top_p, repeat_penalty, stop, tools=None, tool_choice=None):
    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": prepare_lmstudio_messages(system_message, user_message, messages, base64_images),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "presence_penalty": repeat_penalty,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed
    }

    if stop:
        data["stop"] = stop
    if tools:
        data["functions"] = tools
    if tool_choice:
        data["function_call"] = tool_choice

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=data) as response:
                response.raise_for_status()
                response_data = await response.json()

        choices = response_data.get('choices', [])
        if choices:
            choice = choices[0]
            message = choice.get('message', {})
            if "function_call" in message:
                return {
                    "choices": [{
                        "message": {
                            "function_call": {
                                "name": message["function_call"]["name"],
                                "arguments": message["function_call"]["arguments"]
                            }
                        }
                    }]
                }
            else:
                generated_text = message.get('content', '')
                return {
                    "choices": [{
                        "message": {
                            "content": generated_text
                        }
                    }]
                }
        else:
            error_msg = "Error: No valid choices in the LMStudio response."
            print(error_msg)
            return {"choices": [{"message": {"content": error_msg}}]}
    except aiohttp.ClientError as e:
        error_msg = f"Error in LMStudio API request: {e}"
        print(error_msg)
        return {"choices": [{"message": {"content": error_msg}}]}

def prepare_lmstudio_messages(base64_images, system_message, user_message, messages):
    lmstudio_messages = []
    
    if system_message:
        lmstudio_messages.append({"role": "system", "content": system_message})
    
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        if role == "system":
            lmstudio_messages.append({"role": "system", "content": content})
        elif role == "user":
            lmstudio_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            lmstudio_messages.append({"role": "assistant", "content": content})
    
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
        lmstudio_messages.append({
            "role": "user",
            "content": content
        })
        print(f"Number of images sent: {len(base64_images)}")
    else:
        lmstudio_messages.append({"role": "user", "content": user_message})
    
    return lmstudio_messages

  
"""def prepare_lmstudio_messages(system_message, user_message, messages, base64_images=None):
    lmstudio_messages = [
        {"role": "system", "content": system_message},
    ]
    
    for message in messages:
        if isinstance(message["content"], list):
            # Handle multi-modal content
            content = []
            for item in message["content"]:
                if item["type"] == "text":
                    content.append(item["text"])
                elif item["type"] == "image_url":
                    content.append(f"[Image data: {item['image_url']['url']}]")
            lmstudio_messages.append({"role": message["role"], "content": " ".join(content)})
        else:
            lmstudio_messages.append(message)

    if base64_images:
        image_content = "\n".join([f"[Image data: data:image/jpeg;base64,{img}]" for img in base64_images])
        lmstudio_messages.append({
            "role": "user",
            "content": f"{user_message}\n{image_content}"
        })
    else:
        lmstudio_messages.append({"role": "user", "content": user_message})

    return lmstudio_messages"""