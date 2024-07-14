import os
from dotenv import load_dotenv

def get_api_key(api_key_name, engine):
        if engine not in ["ollama", "kobold", "lmstudio", "textgen", "llamacpp", "vllm"]:
            # Try to get the key from .env first
            load_dotenv()
            api_key = os.getenv(api_key_name)
            if api_key:
                return api_key
            else:
                # If .env is empty, get the key from os.environ
                api_key = os.getenv(api_key_name) 
                if api_key:
                    return api_key
                else:
                    raise ValueError(f"{api_key_name} not found. Please set it in your .env file or as an environment variable.")
        else:
            print(f'You are using {engine} as the engine, no API key is required.')
            return None