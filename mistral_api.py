#mistral_api.py
import aiohttp
import asyncio
import json
from typing import List, Union, Optional
import requests
import logging
from mistralai import Mistral
#import base64
#import os

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

async def send_mistral_request(base64_images, model, system_message, user_message, messages, api_key, 
                        seed, temperature, max_tokens, top_p, tools=None, tool_choice=None):
    try:
        client = Mistral(api_key=api_key)   

        # Prepare messages
        mistral_messages = prepare_mistral_messages(base64_images, system_message, user_message, messages)

        #logger.debug(f"Sending messages: {json.dumps(mistral_messages, indent=2)}")

        completion = await client.chat.complete_async(
            model=model,
            messages=mistral_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            random_seed=seed,
            tools=tools,
            tool_choice=tool_choice,
            safe_prompt=False
        )

        #logger.debug(f"Received response: {completion}")

        if tools:
            return completion
        else:
            if hasattr(completion, 'choices') and len(completion.choices) > 0:
                content = completion.choices[0].message.content
                return {
                    "choices": [{
                        "message": {
                            "content": content
                        }
                    }]
                }
            else:
                error_msg = "Error: No valid choices in the MISTRAL response."
                logger.error(error_msg)
                return {"choices": [{"message": {"content": error_msg}}]}

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"choices": [{"message": {"content": f"An unexpected error occurred: {str(e)}"}}]}

def prepare_mistral_messages(base64_images, system_message, user_message, messages):
    mistral_messages = []
    
    if system_message:
        mistral_messages.append({"role": "system", "content": system_message})
    
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        if role == "system":
            mistral_messages.append({"role": "system", "content": content})
        elif role == "user":
            mistral_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            mistral_messages.append({"role": "assistant", "content": content})
    
    # Add the current user message with all images if provided
    if base64_images:
        content = [{"type": "text", "text": user_message}]
        for base64_image in base64_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        mistral_messages.append({
            "role": "user",
            "content": content
        })
        #logger.debug(f"Number of images sent: {len(base64_images)}")
    else:
        mistral_messages.append({"role": "user", "content": user_message})
    
    return mistral_messages

async def create_mistral_compatible_embedding(api_key, model, input):
    try:
        client = Mistral(api_key=api_key)
        embedding = await client.embeddings.create(model=model, input=input)
        
        if hasattr(embedding, 'data') and len(embedding.data) > 0 and hasattr(embedding.data[0], 'embedding'):
            return embedding.data[0].embedding  # Return the embedding directly as a list
        elif hasattr(embedding, 'data') and len(embedding.data) == 0:
            raise ValueError("No embedding generated for the input text.")
        else:
            raise ValueError("Unexpected response format: 'embedding' data not found")
    except Exception as e:
        logger.error(f"Error creating Mistral embedding: {str(e)}")
        raise

