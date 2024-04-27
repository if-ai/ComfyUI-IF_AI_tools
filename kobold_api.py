import requests

def send_kobold_request(endpoint, stop, system_message, user_message, 
                        messages, base64_image, max_length, temperature, 
                        top_k, top_p, rep_pen):
    try:

        prompt = f"System: {system_message}\n"
        for message in messages:
            role = message["role"]
            content = message["content"]
            prompt += f"{role.capitalize()}: {content}\n"
        prompt += f"User: {user_message}" 

        data = {
            "prompt": prompt,
            "max_length": max_length,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "rep_pen": rep_pen,
            "stop_sequence": stop if stop else None
        }

        if base64_image:
            data["images"] = base64_image

        response = requests.post(endpoint, json=data)

        if response.status_code == 200:
            result = response.json()["results"][0]["text"]
            response_text = result.split('\n')[0].replace("  ", " ").strip()
            return response_text
        else:
            return f"Error: {response.status_code} - {response.text}"

    except Exception as e:
        return f"Error: {str(e)}"

