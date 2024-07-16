import requests
import json
def send_ollama_request(endpoint, base64_image, model, system_message, user_message, messages, seed,
                        temperature, max_tokens, random, top_k, top_p, repeat_penalty, stop, keep_alive,
                        tools=None, tool_choice=None):
    try:
        ollama_messages = prepare_ollama_messages(system_message, user_message, messages, base64_image)

        options = {k: v for k, v in {
            "temperature": temperature,
            "num_predict": max_tokens,
            "top_k": top_k,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "stop": stop if stop else None
        }.items() if v is not None}

        if random:
            options["seed"] = seed

        data = {
            "model": model,
            "messages": ollama_messages,
            "stream": False,
            "options": options,
            "keep_alive": -1 if keep_alive else 0,
        }

        # Add tools if provided
        if tools:
            data["tools"] = tools
        if tool_choice:
            data["tool_choice"] = tool_choice

        ollama_headers = {"Content-Type": "application/json"}
        response = requests.post(endpoint, headers=ollama_headers, json=data)
        #print(f"Debugging - Ollama response status: {response.status_code}")
        #print(f"Debugging - Ollama response headers: {response.headers}")

        try:
            response_json = response.json()
            #print(f"Debugging - Ollama response JSON: {json.dumps(response_json, indent=2)}")
            
            if "response" in response_json:
                return response_json["response"].strip()
            elif "message" in response_json:
                return response_json["message"]["content"].strip()
            else:
                return f"Error: Unexpected response format - {json.dumps(response_json)}"
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {str(e)}")
            return f"Error: Failed to decode JSON response - {str(e)}"

    except Exception as e:
        print("Debugging - Exception occurred:")
        print(str(e))
        return f"Error: {str(e)}"

def prepare_ollama_messages(system_message, user_message, messages, base64_image=None):
    ollama_messages = [
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
            ollama_messages.append({"role": message["role"], "content": " ".join(content)})
        else:
            ollama_messages.append(message)

    if base64_image:
        ollama_messages.append({
            "role": "user",
            "content": f"{user_message}\n[Image data: data:image/jpeg;base64,{base64_image}]"
        })
    else:
        ollama_messages.append({"role": "user", "content": user_message})

    return ollama_messages


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