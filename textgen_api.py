import requests

def send_textgen_request(api_url, base64_image, selected_model, system_message, user_message, messages, seed, temperature, max_tokens, top_k, top_p, repeat_penalty, stop):
    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "model": selected_model,
        "messages": prepare_textgen_messages(base64_image, system_message, user_message, messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "presence_penalty": repeat_penalty,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "stop": stop if stop else None
    }

    response = requests.post(api_url, headers=headers, json=data)

    if response.status_code == 200:
        response_data = response.json()
        choices = response_data.get('choices', [])
        if choices:
            choice = choices[0]
            message = choice.get('message', {})
            generated_text = message.get('content', '')
            if generated_text:
                return generated_text
            else:
                print("No content found in the response message.")
        else:
            print("No valid choices in the response.")
    else:
        print(f"Failed to fetch response, status code: {response.status_code}")
        print("Full response:", response.text)

    return "Failed to fetch response from Kobold API."

def prepare_textgen_messages(base64_image, system_message, user_message, messages):
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
            if base64_image:
                textgen_messages.append({
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
                textgen_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            textgen_messages.append({"role": "assistant", "content": content})
    
    return textgen_messages
    
