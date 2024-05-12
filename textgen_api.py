import requests

def send_textgen_request(api_url, selected_model, system_message, user_message, messages, temperature, max_tokens, stop):
    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "model": selected_model,
        "messages": prepare_textgen_messages(system_message, user_message, messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": stop
    }

    response = requests.post(api_url, headers=headers, json=data)

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
        return "Failed to fetch response from text-generation-webui."

def prepare_textgen_messages(system_message, user_message, messages):
    textgen_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        if role == "system":
            textgen_messages.append({"role": "system", "content": content})
        elif role == "user":
            textgen_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            textgen_messages.append({"role": "assistant", "content": content})
    
    return textgen_messages