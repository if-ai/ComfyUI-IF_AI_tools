# gemini_api.py
import aiohttp
import json
import logging
import asyncio

logger = logging.getLogger(__name__)

async def send_gemini_request(base64_images, model, system_message, user_message, messages,
                             temperature, max_tokens, top_k, top_p, stop, api_key,
                             tools=None, tool_choice=None):
    headers = {
        "Content-Type": "application/json"
    }
    base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    # Append the API key to the URL
    api_url = f"{base_url}?key={api_key}"
    
    gemini_messages = prepare_gemini_messages(base64_images, system_message, user_message, messages)
    
    data = {
        "contents": gemini_messages,
        "generationConfig": {
            "temperature": temperature,
            "topP": top_p, 
            "topK": top_k,
            "maxOutputTokens": max_tokens,
            "stopSequences": stop if isinstance(stop, list) else [stop]
        }
    }

    if tools:
        data["generationConfig"]["tools"] = [{"functionDeclarations": tools}] # Changed to functionDeclarations

    if tool_choice:
        data["toolChoice"] = tool_choice  # Assuming Gemini supports this

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=data) as response:
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                response_data = await response.json()


                if tools:
                    return response_data
                else:
                    candidates = response_data.get('candidates', [])
                    if candidates:
                        candidate = candidates[0]
                        content = candidate.get('content', {})
                        if 'parts' in content:
                            for part in content['parts']:
                                if 'functionCall' in part:
                                    return {
                                        "function_call": {
                                            "name": part['functionCall']['name'],
                                            "arguments": json.loads(part['functionCall']['args'])
                                        }
                                    }
                        generated_text = content.get('parts', [{}])[0].get('text', '')
                        return {
                            "choices": [{
                                "message": {
                                    "content": generated_text
                                }
                            }]
                        }
                    else:
                        error_msg = "Error: No valid candidates in the Gemini response."
                        logger.error(error_msg)
                        return {"choices": [{"message": {"content": error_msg}}]}  # Return error in unified format

    except Exception as e:
        error_msg = "Unexpected error during Gemini API call"
        # Log the full error for debugging but return sanitized message
        logger.error(f"{error_msg}: {str(e)}")
        return {"choices": [{"message": {"content": error_msg}}]}
    
def prepare_gemini_messages(base64_images, system_message, user_message, messages):
    gemini_messages = []

    # Add system message if provided
    if system_message:
        gemini_messages.append({"role": "user", "parts": [{"text": f"System: {system_message}"}]})

    # Add previous messages
    for message in messages:
        role = "model" if message["role"] == "assistant" else message["role"]
        content = message["content"]
        
        if isinstance(content, list):
            gemini_messages.append({"role": role, "parts": content})
        else:
            gemini_messages.append({"role": role, "parts": [{"text": content}]})

    # Add current user message with multiple images
    if base64_images:
        parts = [{"text": user_message}]
        for base64_image in base64_images:
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": base64_image
                }
            })
        gemini_messages.append({
            "role": "user",
            "parts": parts
        })
    else:
        gemini_messages.append({"role": "user", "parts": [{"text": user_message}]})
    
    return gemini_messages