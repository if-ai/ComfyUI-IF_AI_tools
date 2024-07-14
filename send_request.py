from .anthropic_api import send_anthropic_request
from .ollama_api import send_ollama_request
from .openai_api import send_openai_request
from .kobold_api import send_kobold_request
from .groq_api import send_groq_request
from .lms_api import send_lmstudio_request
from .textgen_api import send_textgen_request
from .llamacpp_api import send_llama_cpp_request
from .mistral_api import send_mistral_request
from .vllm_api import send_vllm_request
from .gemini_api import send_gemini_request
import json

def send_request(engine, base_ip, port, base64_image, model, system_message, user_message, messages, seed,
                 temperature, max_tokens, random, top_k, top_p, repeat_penalty, stop, keep_alive,
                 api_key=None, tools=None, tool_choice=None):

    api_functions = {
        "groq": send_groq_request,
        "anthropic": send_anthropic_request,
        "openai": send_openai_request,
        "kobold": send_kobold_request,
        "ollama": send_ollama_request,
        "lmstudio": send_lmstudio_request,
        "textgen": send_textgen_request,
        "llamacpp": send_llama_cpp_request,
        "mistral": send_mistral_request,
        "vllm": send_vllm_request,
        "gemini": send_gemini_request,
    }

    if engine not in api_functions:
        raise ValueError(f"Invalid engine: {engine}")

    api_function = api_functions[engine]

    # Prepare a dictionary to store API-specific keyword arguments
    kwargs = {}
    
    if engine == "ollama":
        endpoint = f"http://{base_ip}:{port}/api/chat"  # Changed to /api/chat
        kwargs = {
            "endpoint": endpoint,
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
            "tools": tools,
            "tool_choice": tool_choice,
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
            "tools": tools,
            "tool_choice": tool_choice,
        }
        response = api_function(**kwargs)
    elif engine == "lmstudio":
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
            "tools": tools,
            "tool_choice": tool_choice,
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
            "tools": tools,
            "tool_choice": tool_choice,
        }
        response = api_function(**kwargs)
    elif engine == "llamacpp":
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
    elif engine == "vllm":
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
            "api_key": api_key,
        }
        response = api_function(**kwargs)
    elif engine == "gemini":
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        kwargs = {
            "api_url": api_url,
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
            "api_key": api_key,
            "tools": tools,
            "tool_choice": tool_choice,
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
            "seed": seed if random else None,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "tools": tools,
            "tool_choice": tool_choice,
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
            "tools": tools,
            "tool_choice": tool_choice,
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
            "tools": tools,
            "tool_choice": tool_choice,
        }
        response = api_function(**kwargs)
    elif engine == "mistral":
        kwargs = {
            "api_url": "https://api.mistral.ai/v1/chat/completions",
            "model": model,
            "system_message": system_message,
            "user_message": user_message,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "api_key": api_key,
            "tools": tools,
            "tool_choice": tool_choice,
            "seed": seed if random else None
        }
        response = api_function(**kwargs)
        print(f"Mistral API response: {response}")
        
        try:
            response_json = json.loads(response)
            if isinstance(response_json, dict) and "tool_calls" in response_json:
                # Handle tool calls if needed
                print("Tool calls detected in response")
                return response_json
            else:
                return response
        except json.JSONDecodeError:
            # If it's not JSON, it's a regular text response
            return response
    else:
        raise ValueError(f"Unsupported engine: {engine}")
    
    if isinstance(response, str) and response.startswith("Failed to fetch response"):
            print(f"Error from {engine} API: {response}")
    elif isinstance(response, str) and response.startswith("Error:"):
            print(f"{engine} API request failed. Please check the {engine} server and model.")
    #else:
        #print(f"{engine} API response: {response}")
    
    if tools:
            # If tools are being used, try to parse the response as JSON
        try:
            response_json = json.loads(response)
            return response_json["content"]
        except json.JSONDecodeError:
            print("Failed to parse tool response as JSON")
            return response
    else:
        return response

    
