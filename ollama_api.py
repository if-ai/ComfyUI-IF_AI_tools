#ollama_api.py
import aiohttp
import asyncio
import json
import requests
from typing import List, Union
import logging

logger = logging.getLogger(__name__)
async def create_ollama_embedding(api_base: str, model: str, prompt: Union[str, List[str]]) -> List[float]:
    """
    Create embeddings using Ollama with the REST API asynchronously.
    
    :param api_base: The base URL for the Ollama API
    :param model: The name of the Ollama model to use
    :param prompt: A string or list of strings to embed
    :return: A list of embeddings
    """
    # Normalize the API base URL
    api_base = api_base.rstrip('/')
    if not api_base.endswith('/api'):
        api_base += '/api'
    
    url = f"{api_base}/embeddings"
    
    payload = {
        "model": model,
        "prompt": prompt if isinstance(prompt, str) else prompt[0]  # API expects a single string
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json=payload, headers=headers) as response:
                response.raise_for_status()
                result = await response.json()
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Error calling Ollama embedding API: {str(e)}") from e
    
    if "embedding" in result:
        return result["embedding"]
    else:
        raise ValueError("Unexpected response format: 'embedding' key not found")

async def send_ollama_request(api_url, base64_images, model, system_message, user_message, messages, seed,
                              temperature, max_tokens, random, top_k, top_p, repeat_penalty, stop, keep_alive,
                              tools=None, tool_choice=None):
    """
    Sends a request to the Ollama API and returns a unified response format.
    
    Args:
        api_url (str): The Ollama API endpoint URL.
        base64_images (List[str]): List of images encoded in base64.
        model (str): The model to use.
        system_message (str): System message for the LLM.
        user_message (str): User message for the LLM.
        messages (List[Dict[str, Any]]): Conversation messages.
        seed (int): Random seed.
        temperature (float): Temperature for randomness.
        max_tokens (int): Maximum tokens to generate.
        random (bool): Whether to use randomness.
        top_k (int): Top K for sampling.
        top_p (float): Top P for sampling.
        repeat_penalty (float): Penalty for repetition.
        stop (List[str] or None): Stop sequences.
        keep_alive (bool): Whether to keep the session alive.
        tools (Any, optional): Tools to be used.
        tool_choice (Any, optional): Tool choice.
    
    Returns:
        Union[str, Dict[str, Any]]: Standardized response.
    """
    try:
        ollama_messages = prepare_ollama_messages(system_message, user_message, messages, base64_images)

        options = {
            "num_predict": max_tokens,
            "top_k": top_k,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "stop": stop if stop else None
        }
        options = {k: v for k, v in options.items() if v is not None}

        if random:
            options["seed"] = seed
        else:
            options["temperature"] = temperature
            
        data = {
            "model": model,
            "messages": ollama_messages,
            "stream": False,
            "options": options,
            "keep_alive": -1 if keep_alive else 0,
        }

        # Add tools if provided
        if tools:
            data["tools"] = tools
        if tool_choice:
            data["tool_choice"] = tool_choice

        ollama_headers = {"Content-Type": "application/json"}

        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=ollama_headers, json=data) as response:
                response.raise_for_status()
                response_json = await response.json()

        if tools:
            return response_json
        else:
            if "response" in response_json:
                return {"choices": [{"message": {"content": response_json["response"].strip()}}]}
            elif "message" in response_json:
                return {"choices": [{"message": {"content": response_json["message"]["content"].strip()}}]}
            else:
                error_msg = f"Error: Unexpected response format - {json.dumps(response_json)}"
                logger.error(error_msg)
                return {"choices": [{"message": {"content": error_msg}}]}

    except aiohttp.ClientResponseError as e:
        error_msg = f"HTTP error occurred: {e.status}, message='{e.message}', url={e.request_info.real_url}"
        logger.error(error_msg)
        return {"choices": [{"message": {"content": error_msg}}]}
    except json.JSONDecodeError as e:
        error_msg = f"Error decoding JSON: {str(e)}"
        logger.error(error_msg)
        return {"choices": [{"message": {"content": error_msg}}]}
    except Exception as e:
        error_msg = f"Exception during API call: {str(e)}"
        logger.error(error_msg)
        return {"choices": [{"message": {"content": error_msg}}]}

def prepare_ollama_messages(system_message, user_message, messages, base64_images=None):
    """
    Prepares messages for the Ollama API.

    Args:
        system_message (str): The system message.
        user_message (str): The user message.
        messages (List[Dict[str, Any]]): Previous conversation messages.
        base64_images (List[str], optional): Base64-encoded images.

    Returns:
        List[Dict[str, Any]]: Formatted messages.
    """
    ollama_messages = [
        {"role": "system", "content": system_message},
    ]
    
    for message in messages:
        ollama_messages.append(message)

    if base64_images:
        ollama_messages.append({
            "role": "user",
            "content": user_message,
            "images": base64_images
        })
    else:
        ollama_messages.append({"role": "user", "content": user_message})

    return ollama_messages

def parse_function_call(response, tools):
    try:
        # Look for JSON-like structure in the response
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end != -1:
            json_str = response[start:end]
            parsed = json.loads(json_str)
            if "function_call" in parsed:
                return parsed
    except json.JSONDecodeError:
        pass
    
    return None