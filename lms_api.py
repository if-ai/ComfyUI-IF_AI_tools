import requests

def send_lmstudio_request(api_url, base64_image, model, system_message, user_message, messages, seed, temperature, max_tokens, top_k, top_p, repeat_penalty, stop):
    lmstudio_url = api_url

    data = {
        "model": model,
        "messages": prepare_lmstudio_messages(base64_image, system_message, user_message, messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "presence_penalty": repeat_penalty,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed
    }

    if stop:
        data["stop"] = stop
    

    response = requests.post(lmstudio_url, json=data)

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
        return "Failed to fetch response from LM Studio."

def prepare_lmstudio_messages(base64_image, system_message, user_message, messages):
    lmstudio_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        if role == "system":
            lmstudio_messages.append({"role": "system", "content": content})
        elif role == "user":
            if base64_image:
                lmstudio_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": content
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64," + base64_image,
                                "data": base64_image
                            }
                        }
                    ]
                })
            else:
                lmstudio_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            lmstudio_messages.append({"role": "assistant", "content": content})
    
    return lmstudio_messages