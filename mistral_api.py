import requests
import json

def send_mistral_request(api_url, model, system_message, user_message, messages, temperature, max_tokens, top_p, api_key,
                         tools=None, tool_choice=None, base64_image=None, seed=None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Prepare messages
    if system_message:
        messages = [{"role": "system", "content": system_message}] + messages
    messages.append({"role": "user", "content": user_message})

    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p
    }

    # Add tools if provided
    if tools:
        data["tools"] = tools
    if tool_choice:
        data["tool_choice"] = tool_choice

    # Add seed if provided
    if seed is not None:
        data["random_seed"] = seed

    #print(f"Debugging - Mistral request data: {json.dumps(data, indent=2)}")
    
    response = requests.post(api_url, headers=headers, json=data)
    
    #print(f"Debugging - Mistral response status: {response.status_code}")
    #print(f"Debugging - Mistral response headers: {response.headers}")
    
    if response.status_code == 200:
        response_json = response.json()
        #print(f"Debugging - Mistral response JSON: {json.dumps(response_json, indent=2)}")
        
        # Check for tool calls in the response
        message = response_json["choices"][0]["message"]
        if "tool_calls" in message and message["tool_calls"] is not None:
            tool_calls = message["tool_calls"]
            for tool_call in tool_calls:
                if tool_call["type"] == "function":
                    function_call = tool_call["function"]
                    function_name = function_call["name"]
                    function_args = json.loads(function_call["arguments"])
                    #print(f"Function call: {function_name}")
                    #print(f"Arguments: {function_args}")
            return json.dumps(message)  # Return the entire message including tool calls
        else:
            return message["content"]
    else:
        print(f"Error response: {response.text}")
        return f"Error: {response.status_code} - {response.text}"