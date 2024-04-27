import requests

def send_openai_request(selected_model, system_message, user_message, messages, api_key, temperature, max_tokens, base64_image):
    openai_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": selected_model,
        "messages": prepare_openai_messages(base64_image, system_message, user_message, messages),
        "temperature": temperature,
        "max_tokens": max_tokens
    }

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
        return "Failed to fetch response from OpenAI."

def prepare_openai_messages(base64_image, system_message, user_message, messages):
    openai_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        if role == "system":
            openai_messages.append({"role": "system", "content": content})
        elif role == "user":
            if base64_image:
                openai_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": content
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{base64_image}"
                        }
                    ]
                })
            else:
                openai_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            openai_messages.append({"role": "assistant", "content": content})
    
    return openai_messages



