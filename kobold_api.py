#kobold_api.py
import aiohttp
import asyncio
import json
import logging
from typing import List, Union, Optional, Dict, Any

logger = logging.getLogger(__name__)

async def send_kobold_request(api_url, base64_images, model, system_message, user_message, messages, seed,
                              temperature, max_tokens, top_k, top_p, repeat_penalty, stop,
                              tools=None, tool_choice=None):
    """
    Sends an asynchronous request to the Kobold API and returns a unified response format.
    
    Args:
        api_url (str): The Kobold API endpoint URL.
        base64_images (List[str]): List of images encoded in base64.
        model (str): The model to use.
        system_message (str): System message for the LLM.
        user_message (str): User message for the LLM.
        messages (List[Dict[str, Any]]): Conversation messages.
        seed (int): Random seed.
        temperature (float): Temperature for randomness.
        max_tokens (int): Maximum tokens to generate.
        top_k (int): Top K for sampling.
        top_p (float): Top P for sampling.
        repeat_penalty (float): Penalty for repetition.
        stop (List[str] or None): Stop sequences.
        tools (Any, optional): Tools to be used.
        tool_choice (Any, optional): Tool choice.
    
    Returns:
        Union[str, Dict[str, Any]]: Standardized response.
    """
    try:
        kobold_messages = prepare_kobold_messages(base64_images, system_message, user_message, messages)

        headers = {
            "Content-Type": "application/json"
        }

        data = {
            "model": model,
            "messages": kobold_messages,
            "max_length": max_tokens,
            "rep_pen": repeat_penalty,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "seed": seed
        }

        if stop:
            data["stop_sequence"] = stop

        if tools:
            data["tools"] = tools
        if tool_choice:
            data["tool_choice"] = tool_choice

        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=data) as response:
                response.raise_for_status()
                response_json = await response.json()

        if tools:
            return response_json
        else:
            try:
                content = extract_content(response_json, tools is not None)
                if content:
                    return {"choices": [{"message": {"content": content}}]}
                else:
                    error_msg = "Error: No content found in response."
                    logger.error(error_msg)
                    return {"choices": [{"message": {"content": error_msg}}]}
            except json.JSONDecodeError as e:
                error_msg = f"Error decoding JSON: {str(e)}"
                logger.error(error_msg)
                return {"choices": [{"message": {"content": error_msg}}]}

    except aiohttp.ClientError as e:
        error_msg = f"HTTP error occurred: {str(e)}"
        logger.error(error_msg)
        return {"choices": [{"message": {"content": error_msg}}]}
    except Exception as e:
        error_msg = f"Exception during Kobold API call: {str(e)}"
        logger.error(error_msg)
        return {"choices": [{"message": {"content": error_msg}}]}

def extract_content(response_json: Dict[str, Any], is_tool_response: bool) -> Optional[str]:
    """
    Extracts content from the Kobold API response.

    Args:
        response_json (Dict[str, Any]): The raw response from the API.
        is_tool_response (bool): Indicates if tools were used.

    Returns:
        Optional[str]: Extracted content or None.
    """
    try:
        choices = response_json.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            
            if is_tool_response:
                # For tool responses, return the entire content as JSON
                return json.dumps({"content": content})
            else:
                # For normal responses, return the content
                return content
    except Exception as e:
        logger.error(f"Error extracting content: {str(e)}")
    return None

def prepare_kobold_messages(base64_images, system_message, user_message, messages):
    """
    Prepares messages for the Kobold API.

    Args:
        base64_images (List[str]): Base64-encoded images.
        system_message (str): System message.
        user_message (str): User message.
        messages (List[Dict[str, Any]]): Previous conversation messages.

    Returns:
        List[Dict[str, Any]]: Formatted messages.
    """
    kobold_messages = []
    
    if system_message:
        kobold_messages.append({"role": "system", "content": system_message})
    
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        if role in ["system", "user", "assistant"]:
            kobold_messages.append({"role": role, "content": content})
    
    # Add the current user message with image if provided
    if base64_images:
        content = [{"type": "text", "text": user_message}]
        for img in base64_images:
            content.append({
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{img}"
            })
        kobold_messages.append({"role": "user", "content": content})
    else:
        kobold_messages.append({"role": "user", "content": user_message})
    
    return kobold_messages

