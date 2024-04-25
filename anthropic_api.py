import requests

def send_anthropic_request(selected_model, base64_image, system_message, user_message, chat_history, anthropic_api_key, temperature, max_tokens):
    anthropic_headers = {
        "x-api-key": anthropic_api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }

    data = {
        "model": selected_model,
        "messages": prepare_anthropic_messages(base64_image, system_message, user_message, chat_history),
        "temperature": temperature,
        "max_tokens_to_sample": max_tokens
    }

    api_url = 'https://api.anthropic.com/v1/completions'
    response = requests.post(api_url, headers=anthropic_headers, json=data)

    if response.status_code == 200:
        response_data = response.json()
        generated_text = response_data.get('completion', '')
        return generated_text.strip()
    else:
        print(f"Error: Request failed with status code {response.status_code}, Response: {response.text}")
        return "Failed to fetch response from Anthropic."

def prepare_anthropic_messages(base64_image, system_message, user_message, chat_history):
    messages = [
        {
            "role": "system",
            "content": system_message
        }
    ]
    
    for message in chat_history:
        role = message["role"]
        content = message["content"]
        messages.append({"role": role, "content": content})
    
    if base64_image:
        messages.append({
            "role": "user",
            "content": f"{user_message}\n\nImage: data:image/png;base64,{base64_image}"
        })
    else:
        messages.append({
            "role": "user",
            "content": user_message
        })
        
    return messages
