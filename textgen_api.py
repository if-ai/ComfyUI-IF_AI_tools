import requests
import json

def send_textgen_request(api_url, base64_image, model, system_message, user_message, messages, seed, 
                         temperature, max_tokens, top_k, top_p, repeat_penalty, stop,
                         tools=None, tool_choice=None):
    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": prepare_textgen_messages(system_message, user_message, messages, base64_image),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_k": top_k,
        "top_p": top_p,
        "repeat_penalty": repeat_penalty,
        "stop": stop,
        "seed": seed
    }

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
        print(f"Error in textgen API request: {e}")
        return str(e)

def prepare_textgen_messages(system_message, user_message, messages, base64_image=None):
    textgen_messages = []
    
    if system_message:
        textgen_messages.append({"role": "system", "content": system_message})
    
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        if isinstance(content, list):
            # Handle multi-modal content
            message_content = []
            for item in content:
                if item["type"] == "text":
                    message_content.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image_url":
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": item["image_url"]["url"]}
                    })
            textgen_messages.append({"role": role, "content": message_content})
        else:
            textgen_messages.append({"role": role, "content": content})

    # Add the current user message with image if provided
    if base64_image:
        textgen_messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_message},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        })
    else:
        textgen_messages.append({"role": "user", "content": user_message})

    return textgen_messages

def parse_function_call(response, tools):
    try:
        # Look for JSON-like structure in the response
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end != -1:
            json_str = response[start:end]
            parsed = json.loads(json_str)
            if "function_call" in parsed:
                return parsed
    except json.JSONDecodeError:
        pass
    
    return None