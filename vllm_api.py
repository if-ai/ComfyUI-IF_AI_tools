import requests
import json

def send_vllm_request(api_url, base64_image, model, system_message, user_message, messages, seed, 
                      temperature, max_tokens, top_k, top_p, repeat_penalty, stop, api_key,
                      tools=None, tool_choice=None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": model,
        "messages": prepare_vllm_messages(system_message, user_message, messages, base64_image),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repeat_penalty,
        "stop": stop,
        "seed": seed
    }

    if tools:
        data["functions"] = tools
    if tool_choice:
        if tool_choice == "auto":
            data["function_call"] = "auto"
        elif tool_choice == "none":
            data["function_call"] = "none"
        else:
            data["function_call"] = {"name": tool_choice["function"]["name"]}

    response = requests.post(api_url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        response_data = response.json()
        message = response_data["choices"][0]["message"]
        
        if "function_call" in message:
            return {
                "function_call": {
                    "name": message["function_call"]["name"],
                    "arguments": message["function_call"]["arguments"]
                }
            }, messages
        else:
            return message["content"], messages
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

def prepare_vllm_messages(system_message, user_message, messages, base64_image=None):
    vllm_messages = [
        {"role": "system", "content": system_message},
    ]
    
    for message in messages:
        vllm_messages.append(message)

    if base64_image:
        vllm_messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_message},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        })
    else:
        vllm_messages.append({"role": "user", "content": user_message})

    return vllm_messages