from .anthropic_api import send_anthropic_request
from .ollama_api import send_ollama_request
from .openai_api import send_openai_request
from .kobold_api import send_kobold_request
from .groq_api import send_groq_request
from .lms_api import send_lmstudio_request
from .textgen_api import send_textgen_request

def send_request(engine, base_ip, port, base64_image, model, system_message, user_message, messages, seed, 
                 temperature, max_tokens, random, top_k, top_p, repeat_penalty, stop, keep_alive, 
                 chat_history, api_key=None):
    api_functions = {
        "groq": send_groq_request,
        "anthropic": send_anthropic_request,
        "openai": send_openai_request,
        "kobold": send_kobold_request,
        "ollama": send_ollama_request,
        "lms": send_lmstudio_request,
        "textgen": send_textgen_request
    }

    if engine not in api_functions:
        raise ValueError(f"Invalid engine: {engine}")

    api_function = api_functions[engine]

    # Prepare a dictionary to store API-specific keyword arguments
    kwargs = {}
    
    if engine == "ollama":
        kwargs = {
            "endpoint": f"http://{base_ip}:{port}/api/generate",
            "base64_image": base64_image,
            "model": model,
            "system_message": system_message,
            "user_message": user_message,
            "messages": messages,
            "seed": seed,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "random": random,
            "top_k": top_k,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "stop": stop,
            "keep_alive": keep_alive,
        }
        response = api_function(**kwargs)
    elif engine == "kobold":
        kwargs = {
            "api_url": f"http://{base_ip}:{port}/v1/chat/completions",
            "base64_image": base64_image,
            "model": model,
            "system_message": system_message,
            "user_message": user_message,
            "messages": messages,
            "seed": seed,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_k": top_k,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "stop": stop,
        }
        response = api_function(**kwargs)
    elif engine == "lms":
        kwargs = {
            "api_url": f"http://{base_ip}:{port}/v1/chat/completions",
            "base64_image": base64_image,
            "model": model,
            "system_message": system_message,
            "user_message": user_message,
            "messages": messages,
            "seed": seed,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_k": top_k,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "stop": stop,
        }
        response = api_function(**kwargs)
    elif engine == "textgen":
        kwargs = {
            "api_url": f"http://{base_ip}:{port}/v1/chat/completions",
            "base64_image": base64_image,
            "model": model,
            "system_message": system_message,
            "user_message": user_message,
            "messages": messages,
            "seed": seed,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_k": top_k,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "stop": stop,
        }
        response = api_function(**kwargs)
    elif engine == "openai":
        kwargs = {
            "base64_image": base64_image,
            "model": model,
            "system_message": system_message,
            "user_message": user_message,
            "messages": messages,
            "api_key": api_key,
            "seed": seed,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
        }
        response = api_function(**kwargs)
    elif engine == "anthropic":
        kwargs = {
            "model": model,
            "system_message": system_message,
            "user_message": user_message,
            "messages": messages,
            "api_key": api_key,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "base64_image": base64_image,
        }
        response = api_function(**kwargs)
    elif engine == "groq":
        kwargs = {
            "model": model,
            "system_message": system_message,
            "user_message": user_message,
            "messages": messages,
            "api_key": api_key,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "base64_image": base64_image,
        }
        response = api_function(**kwargs)
    else:
        raise ValueError(f"Unsupported engine: {engine}")

    # Update chat history
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": response})

    return response, chat_history