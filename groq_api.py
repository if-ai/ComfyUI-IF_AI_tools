from groq import Groq
import json

def send_groq_request(model, system_message, user_message, 
                      messages, api_key, temperature, max_tokens, 
                      base64_image=None):
    try:
        client = Groq(api_key=api_key)

        data = {
            "model": model,
            "messages": prepare_groq_messages(base64_image, system_message, user_message, messages),
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        response = client.chat.completions.create(**data)

        if response.choices:
            choice = response.choices[0]
            message = choice.message

            try:
                assistant_content = json.loads(message.content)[0].get('content')
            except json.JSONDecodeError:
                assistant_content = message.content 

            assistant_content = assistant_content.replace("\n\n", " ").strip()


            print(assistant_content)
            return assistant_content
           
        else:
            print("No valid choices in the response.")
            print("Full response:", response)
            return "No valid response generated.", messages

    except Exception as e:
        return f"Error: {str(e)}", messages

def prepare_groq_messages(base64_image, system_message, user_message, messages):
    # Add the user's message to the history
    messages.append({"role": "user", "content": user_message})

    groq_messages = [{"role": "system", "content": system_message}]

    for message in messages:
        if message["role"] != "system":
            # Removed str conversion 
            groq_messages.append(message)

    if base64_image:
        groq_messages[1]["image"] = base64_image

    return groq_messages
