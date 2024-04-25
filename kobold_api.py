import requests
def send_kobold_request(user_message, system_message, endpoint, stop, messages, base64_image, max_length, temperature, top_k, top_p, rep_pen):
    try:
        conversation_history = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            conversation_history += f"{role}: {content}\n"

        prompt = f"{system_message}\n{conversation_history}User: {user_message}\nBot:"

        data = {
            "prompt": prompt,
            "max_length": max_length,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "rep_pen": rep_pen,
            "stop_sequence": ["\n", f"{stop}"]
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
