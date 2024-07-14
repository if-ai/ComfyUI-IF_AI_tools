import requests
import json

def send_lmstudio_request(api_url, base64_image, model, system_message, user_message, messages, seed, temperature, 
                          max_tokens, top_k, top_p, repeat_penalty, stop, tools=None, tool_choice=None):
    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": prepare_lmstudio_messages(system_message, user_message, messages, base64_image),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "presence_penalty": repeat_penalty,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed
    }

    if stop:
        data["stop"] = stop
    if tools:
        data["functions"] = tools
    if tool_choice:
        data["function_call"] = tool_choice

    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()  # This will raise an exception for HTTP errors
        
        response_data = response.json()
        message = response_data["choices"][0]["message"]
        
        if "function_call" in message:
            return {
                "function_call": {
                    "name": message["function_call"]["name"],
                    "arguments": message["function_call"]["arguments"]
                }
            }
        else:
            return message["content"]
    except requests.exceptions.RequestException as e:
        print(f"Error in LMStudio API request: {e}")
        return str(e)

def prepare_lmstudio_messages(system_message, user_message, messages, base64_image=None):
    lmstudio_messages = [
        {"role": "system", "content": system_message},
    ]
    
    for message in messages:
        if isinstance(message["content"], list):
            # Handle multi-modal content
            content = []
            for item in message["content"]:
                if item["type"] == "text":
                    content.append(item["text"])
                elif item["type"] == "image_url":
                    content.append(f"[Image data: {item['image_url']['url']}]")
            lmstudio_messages.append({"role": message["role"], "content": " ".join(content)})
        else:
            lmstudio_messages.append(message)

    if base64_image:
        lmstudio_messages.append({
            "role": "user",
            "content": f"{user_message}\n[Image data: data:image/jpeg;base64,{base64_image}]"
        })
    else:
        lmstudio_messages.append({"role": "user", "content": user_message})

    return lmstudio_messages