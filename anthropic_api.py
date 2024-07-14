import requests

def send_anthropic_request(model, system_message, user_message, messages, api_key, temperature, max_tokens, base64_image, tools=None, tool_choice=None):
    
    anthropic_headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    data = {
        "system": system_message,
        "model": model,
        "messages": prepare_anthropic_messages(user_message, messages, base64_image=base64_image),
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    if tools:
        data["tools"] = tools
    if tool_choice:
        data["tool_choice"] = tool_choice

    api_url = 'https://api.anthropic.com/v1/messages'
    response = requests.post(api_url, headers=anthropic_headers, json=data)
    if response.status_code == 200:
        response_data = response.json()
        content_blocks = response_data.get('content', [])
        if content_blocks:
            generated_text = content_blocks[0].get('text', '')
        else:
            generated_text = ''
        return generated_text.strip()
    else:
        print(f"Error: Request failed with status code {response.status_code}, Response: {response.text}")
        return "Failed to fetch response from Anthropic."

def prepare_anthropic_messages(user_message, messages, base64_image=None):
    anthropic_messages = []

    for message in messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            if anthropic_messages and anthropic_messages[-1]["role"] == "user":
                anthropic_messages[-1]["content"] += f"\n{content}"
            else:
                anthropic_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            anthropic_messages.append({"role": "assistant", "content": content})
    
    if user_message:
        if anthropic_messages and anthropic_messages[-1]["role"] == "user":
            anthropic_messages[-1]["content"] += f"\n{user_message}"
        else:
            anthropic_messages.append({"role": "user", "content": user_message})
    
    for message in anthropic_messages:
        if isinstance(message["content"], str):
            message["content"] = [{"type": "text", "text": message["content"]}]

    if base64_image:
        if anthropic_messages and anthropic_messages[-1]["role"] == "user":
            anthropic_messages[-1]["content"].append({
                "type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_image}
            })
        else:
            anthropic_messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_image}}
                ]
            })

    return anthropic_messages


