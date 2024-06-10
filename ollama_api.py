import requests

def send_ollama_request(endpoint, base64_image, model, system_message, user_message, messages, seed, 
                        temperature, max_tokens, random, top_k, top_p, repeat_penalty, stop, keep_alive):
    try:
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "user":
                prompt += f"User: {content}\n"
            else:
                prompt += f"Assistant: {content}\n"

        data = {
            "model": model,
            "system": system_message,
            "prompt": prompt + f"User: {user_message}\n",
            "stream": False,
            "images": [base64_image] if base64_image else None,
            "options": {
                "temperature": temperature,
                "num_ctx": max_tokens,
                "top_k": top_k,
                "top_p": top_p,
                "repeat_penalty": repeat_penalty,
                "stop": stop if stop else None
            },
            "keep_alive": -1 if keep_alive else 0,
        }

        if random:
            data["options"].setdefault("seed", seed)


        ollama_headers = {"Content-Type": "application/json"}
        response = requests.post(endpoint, headers=ollama_headers, json=data)

        if response.status_code == 200:
            response_json = response.json()
            prompt_response = response_json.get("response", "").strip()
            return prompt_response
        else:
            print("Debugging - Error response:")
            print(response.text)
            return f"Error: {response.status_code} - {response.text}", None

    except Exception as e:
        print("Debugging - Exception occurred:")
        print(str(e))
        return f"Error: {str(e)}", None

