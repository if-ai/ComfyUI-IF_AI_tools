from groq import Groq

def send_groq_request(selected_model, groq_api_key, system_message, user_message, chat_history, temperature, max_tokens, seed, random, base64_image=None):
    try:
        client = Groq(api_key = groq_api_key)
        messages = [{"role": "system", "content": system_message}]
        for message in chat_history:
            messages.append({"role": message["role"], "content": message["content"]})

        if base64_image:
            messages.append({"role": "user", "content": user_message, "image": base64_image})
        else:
            messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model=selected_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed if random else None
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error: {str(e)}"