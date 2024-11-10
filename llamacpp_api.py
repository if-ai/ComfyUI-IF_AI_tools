#llamacpp_api.py
import requests
import json
import base64
import aiohttp
import logging
from typing import List, Union, Optional, Dict, Any
import base64
import os
logger = logging.getLogger(__name__)

async def send_llama_cpp_request(api_url, base64_images, model, system_message, user_message, messages, seed, 
                           temperature, max_tokens, top_k, top_p, repeat_penalty, stop, tools=None):
    headers = {
        "Content-Type": "application/json"
    }

    #api_url = f"{api_url}/v1/chat/completions"
    
    data = {
        "model": model,
        "messages": prepare_llama_cpp_messages(system_message, user_message, messages, base64_images),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_k": top_k,
        "top_p": top_p,
        "frequency_penalty": repeat_penalty,
        "stop": stop,
        "seed": seed
    }
    

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=data) as response:
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
                    
    except requests.exceptions.RequestException as e:
        print(f"Error in LLaMa.cpp API request: {e}")
        return str(e)

def prepare_llama_cpp_messages(system_message, user_message, messages, base64_images=None):
    llama_messages = []
    
    if system_message:
        llama_messages.append({"role": "system", "content": system_message})
    
    for message in messages:
        llama_messages.append(message)

    if base64_images:
        content = [{"type": "text", "text": user_message}]
        for i, img in enumerate(base64_images):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}"}
            })
        llama_messages.append({"role": "user", "content": content})
    else:
        llama_messages.append({"role": "user", "content": user_message})
    
    return llama_messages