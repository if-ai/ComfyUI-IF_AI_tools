import requests
import json

from .get_api import get_api_key

def get_models(engine, base_ip, port):
        
        if engine == "ollama":
            api_url = f'http://{base_ip}:{port}/api/tags'
            try:
                response = requests.get(api_url)
                response.raise_for_status()
                models = [model['name'] for model in response.json().get('models', [])]
                return models
            except Exception as e:
                print(f"Failed to fetch models from Ollama: {e}")
                return []
            
        elif engine == "lmstudio":
            api_url = f'http://{base_ip}:{port}/v1/models'
            try:
                response = requests.get(api_url)
                if response.status_code == 200:
                    data = response.json()
                    models = [model['id'] for model in data['data']]
                    return models
                else:
                    print(f"Failed to fetch models from LM Studio. Status code: {response.status_code}")
                    return []
            except requests.exceptions.RequestException as e:
                print(f"Error connecting to LM Studio server: {e}")
                return []
            
        elif engine == "textgen":
            api_url = f'http://{base_ip}:{port}/v1/internal/model/list'
            try:
                response = requests.get(api_url)
                response.raise_for_status()
                models = response.json()['model_names']
                return models
            except Exception as e:
                print(f"Failed to fetch models from text-generation-webui: {e}")
                return []
            
        elif engine == "kobold":
            api_url = f'http://{base_ip}:{port}/api/v1/model'
            try:
                response = requests.get(api_url)
                response.raise_for_status()
                model = response.json()['result']
                return [model]
            except Exception as e:
                print(f"Failed to fetch models from Kobold: {e}")
                return []
        
        elif engine == "llamacpp":
            api_url = f'http://{base_ip}:{port}/v1/models'
            try:
                response = requests.get(api_url)
                response.raise_for_status()
                models = [model["id"] for model in response.json()["data"]]
                return models
            except Exception as e:
                print(f"Failed to fetch models from llama.cpp: {e}")
                return []

        elif engine == "vllm":
            api_url = f"http://{base_ip}:{port}/v1/models"
            try:
                response = requests.get(api_url)
                response.raise_for_status()
                # Adapt this based on vLLM's actual API response structure 
                models = [model['id'] for model in response.json()['data']] 
                return models
            except Exception as e:
                print(f"Failed to fetch models from vLLM: {e}")
                return []
        
        elif engine == "openai":
            api_key = get_api_key("OPENAI_API_KEY", engine)
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            api_url = "https://api.openai.com/v1/models"
            try:
                response = requests.get(api_url, headers=headers)
                response.raise_for_status()
                models = [model["id"] for model in response.json()["data"]]
                return models
            except Exception as e:
                print(f"Failed to fetch models from OpenAI: {e}")
                return []
        
        elif engine == "mistral":
            api_key = get_api_key("MISTRAL_API_KEY", engine)
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            api_url = "https://api.mistral.ai/v1/models"
            try:
                response = requests.get(api_url, headers=headers)
                response.raise_for_status()
                models = [model["id"] for model in response.json()["data"]]
                return models
            except Exception as e:
                print(f"Failed to fetch models from Mistral: {e}")
                return []

        elif engine == "groq":
            api_key = get_api_key("GROQ_API_KEY", engine)
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            api_url = "https://api.groq.com/openai/v1/models"
            try:
                response = requests.get(api_url, headers=headers)
                response.raise_for_status()
                models = [model["id"] for model in response.json()["data"]]
                return models
            except Exception as e:
                print(f"Failed to fetch models from GROQ: {e}")
                return []
        
        
        elif engine == "anthropic":
            return ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
        
        elif engine == "gemini":
            return ["gemini-1.5-flash", "gemini-1.5-pro-latest", "gemini-1.5-latest", "gemini-pro", "gemini-pro-vision"] 
        
        else:
            print(f"Unsupported engine - {engine}")
            return []
        
