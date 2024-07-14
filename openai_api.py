import requests
import json
import base64

def send_openai_request(base64_image, model, system_message, user_message, messages, api_key, 
                        seed, temperature, max_tokens, top_p, repeat_penalty, tools=None, tool_choice=None):
    openai_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Prepare messages
    openai_messages = prepare_openai_messages(base64_image, system_message, user_message, messages)

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

    api_url = 'https://api.openai.com/v1/chat/completions'
    response = requests.post(api_url, headers=openai_headers, json=data)

    if response.status_code == 200:
        response_data = response.json()
        choices = response_data.get('choices', [])
        if choices:
            choice = choices[0]
            message = choice.get('message', {})
            generated_text = message.get('content', '')
            return generated_text
        else:
            print("No valid choices in the response.")
            print("Full response:", response.text)
            return "No valid response generated."
    else:
        print(f"Failed to fetch response, status code: {response.status_code}")
        print("Full response:", response.text)
        return f"Failed to fetch response from OpenAI. Status code: {response.status_code}"

def prepare_openai_messages(base64_image, system_message, user_message, messages):
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
    
    # Add the current user message with image if provided
    if base64_image:
        openai_messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_message},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        })
    else:
        openai_messages.append({"role": "user", "content": user_message})
    
    return openai_messages
