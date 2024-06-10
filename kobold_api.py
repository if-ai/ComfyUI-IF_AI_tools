import requests
def send_kobold_request(api_url, base64_image, model, system_message, user_message, messages, seed, temperature, max_tokens, top_k, top_p, repeat_penalty, stop):
    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": prepare_kobold_messages(system_message, user_message, messages, base64_image),
        "max_length": max_tokens,
        "rep_pen": repeat_penalty,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "seed": seed
    }

    if stop:
        data["stop_sequence"] = stop

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

def prepare_kobold_messages(system_message, user_message, messages, base64_image=None):
    kobold_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        if role == "system":
            kobold_messages.append({"role": "system", "content": content})
        elif role == "user":
            if base64_image:
                kobold_messages.append({
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
                kobold_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            kobold_messages.append({"role": "assistant", "content": content})
    
    return kobold_messages