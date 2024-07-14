import requests
import json
import base64

import requests
import json

def send_gemini_request(api_url, base64_image, model, system_message, user_message, messages, seed,
                        temperature, max_tokens, top_k, top_p, repeat_penalty, stop, api_key,
                        tools=None, tool_choice=None):
    headers = {
        "Content-Type": "application/json"
    }
    
    # Append the API key to the URL
    api_url = f"{api_url}?key={api_key}"
    
    gemini_messages = prepare_gemini_messages(base64_image, system_message, user_message, messages)
    
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
        data["tools"] = [{"functionDeclarations": tools}]
    
    if tool_choice:
        data["toolChoice"] = tool_choice

    #print(f"Sending request to Gemini API: {api_url}")
    response = requests.post(api_url, headers=headers, json=data)

    if response.status_code == 200:
        response_data = response.json()
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
            return generated_text
        else:
            print("No valid candidates in the response.")
            print("Full response:", response.text)
            return "No valid response generated."
    else:
        print(f"Failed to fetch response, status code: {response.status_code}")
        print("Full response:", response.text)
        return f"Failed to fetch response from Gemini. Status code: {response.status_code}"

    
def prepare_gemini_messages(base64_image, system_message, user_message, messages):
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

    # Add current user message
    if base64_image:
        gemini_messages.append({
            "role": "user",
            "parts": [
                {"text": user_message},
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64_image
                    }
                }
            ]
        })
    else:
        gemini_messages.append({"role": "user", "parts": [{"text": user_message}]})
    
    return gemini_messages