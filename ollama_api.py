import requests

def send_ollama_request(endpoint, base64_image, selected_model, messages, max_tokens, temperature, seed, random, keep_alive, top_k, top_p, repeat_penalty, stop):
    try:
        data = {
            "model": selected_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": max_tokens,
                "top_k": top_k,
                "top_p": top_p,
                "repeat_penalty": repeat_penalty,
            },
            "keep_alive": keep_alive
        }
        if random:
            data["options"].setdefault("seed", seed)

        if base64_image:
            data["images"] = [base64_image]
        
        if stop is not None:
            data["options"]["stop"] = stop

        ollama_headers = {"Content-Type": "application/json"}
        response = requests.post(endpoint, headers=ollama_headers, json=data)

        if response.status_code == 200:
            response_data = response.json()
            prompt_response = response_data.get('message', {}).get('content', 'No response text found').strip()
            return prompt_response
        else:
            return f"Error: {response.status_code} - {response.text}"

    except Exception as e:
        return f"Error: {str(e)}"


