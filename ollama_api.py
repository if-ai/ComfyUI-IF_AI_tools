import requests
import json
import re

def send_ollama_request(endpoint, base64_image, selected_model, system_message, user_message, messages, temperature, max_tokens, seed, random, keep_alive, top_k, top_p, repeat_penalty, stop):
    try:
        prompt = f"System: {system_message}\n"
        for message in messages:
            role = message["role"]
            content = message["content"]
            prompt += f"{role.capitalize()}: {content}\n"
        prompt += f"User: {user_message}"

        data = {
            "model": selected_model,
            "prompt": prompt,
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
            "keep_alive": keep_alive
        }

        if random:
            data["options"].setdefault("seed", seed)



        ollama_headers = {"Content-Type": "application/json"}
        response = requests.post(endpoint, headers=ollama_headers, json=data)


        if response.status_code == 200:
            response_text = response.text
            # Extract the response text using regular expression
            match = re.search(r'"response"\s*:\s*"(.*?)","done"', response_text, re.DOTALL)
            if match:
                prompt_response = match.group(1)
                # Replace escaped newline characters with actual newlines
                prompt_response = prompt_response.replace("\\n", "\n")
                return prompt_response.strip()
            else:
                return "No response text found"
        else:
            return f"Error: {response.status_code} - {response.text}"

    except Exception as e:
        return f"Error: {str(e)}"
    

