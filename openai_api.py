import requests

def send_openai_request(selected_model, base64_image, system_message, user_message, openai_api_key, chat_history, temperature, max_tokens, seed, random):
    openai_headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": selected_model,
        "messages": prepare_openai_messages(base64_image, system_message, user_message, chat_history),
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    if random:
        data["seed"] = seed

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

def prepare_openai_messages(base64_image, system_message, user_message, chat_history):
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
            "content": [
                {
                    "type": "text",
                    "text": user_message
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{base64_image}"
                }
            ]
        })
    else:
        messages.append({
            "role": "user",
            "content": user_message
        })

    return messages



