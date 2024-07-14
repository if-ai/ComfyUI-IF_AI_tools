import requests
import json

def send_kobold_request(api_url, base64_image, model, system_message, user_message, messages, seed,
                        temperature, max_tokens, top_k, top_p, repeat_penalty, stop,
                        tools=None, tool_choice=None):
    try:
        headers = {
            "Content-Type": "application/json"
        }

        kobold_messages = prepare_kobold_messages(base64_image, system_message, user_message, messages)

        data = {
            "model": model,
            "messages": kobold_messages,
            "max_length": max_tokens,
            "rep_pen": repeat_penalty,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "seed": seed
        }

        if stop:
            data["stop_sequence"] = stop if stop else None

        if tools:
            data["tools"] = tools
        if tool_choice:
            data["tool_choice"] = tool_choice

        response = requests.post(api_url, headers=headers, json=data)

        #print(f"Debugging - kobold response status: {response.status_code}")
        #print(f"Debugging - kobold response headers: {response.headers}")

        try:
            response_json = response.json()
            #print(f"Debugging - kobold response JSON: {json.dumps(response_json, indent=2)}")

            # Extract the content from the nested structure
            content = extract_content(response_json, tools is not None)
            return content
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {str(e)}")
            return f"Error: Failed to decode JSON response - {str(e)}"

    except Exception as e:
        print("Debugging - Exception occurred:")
        print(str(e))
        return f"Error: {str(e)}"

def extract_content(response_json, is_tool_response):
    try:
        choices = response_json.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            
            if is_tool_response:
                # For tool responses, return the entire content as JSON
                return json.dumps({"content": content})
            else:
                # For normal responses, try to extract meaningful text
                try:
                    content_json = json.loads(content)
                    if "caption" in content_json:
                        captions = content_json["caption"]
                        if captions and isinstance(captions[0], list):
                            return captions[0][0]  # Return the first caption
                except json.JSONDecodeError:
                    pass
                
                return content
    except Exception as e:
        print(f"Error extracting content: {str(e)}")
    return None

def prepare_kobold_messages(base64_image, system_message, user_message, messages):
    kobold_messages = []
    
    if system_message:
        kobold_messages.append({"role": "system", "content": system_message})
    
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        if role == "system":
            kobold_messages.append({"role": "system", "content": content})
        elif role == "user":
            kobold_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            kobold_messages.append({"role": "assistant", "content": content})
    
    # Add the current user message with image if provided
    if base64_image:
        kobold_messages.append({
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
        kobold_messages.append({"role": "user", "content": user_message})
    
    return kobold_messages

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