import json
import requests
import base64
import textwrap
import io
import os
from io import BytesIO
from PIL import Image
import torch
import tempfile
from torchvision.transforms.functional import to_pil_image
import folder_paths
import anthropic
import openai
from server import PromptServer
from aiohttp import web

@PromptServer.instance.routes.post("/IF_ChatPrompt/get_models")
async def get_models_endpoint(request):
    data = await request.json()
    engine = data.get("engine")
    base_ip = data.get("base_ip")
    port = data.get("port")

    node = IFChatPrompt()
    models = node.get_models(engine, base_ip, port)
    return web.json_response(models)

class IFChatPrompt:

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("Question", "Response",)
    FUNCTION = "describe_picture"
    OUTPUT_NODE = True
    CATEGORY = "ImpactFrames💥🎞️"

    @classmethod
    def INPUT_TYPES(cls):
        node = cls()
        return {
            "required": {
                "image_prompt": ("STRING", {"multiline": True, "default": ""}),
                "base_ip": ("STRING", {"default": node.base_ip}),
                "port": ("STRING", {"default": node.port}),
                "engine": (["ollama", "openai", "anthropic"], {"default": node.engine}),
                #"selected_model": (node.get_models("node.engine", node.base_ip, node.port), {}), 
                "selected_model": ((), {}),
                "profile": ([name for name in node.profiles.keys()], {}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
            },
            "optional": {
                "image": ("IMAGE", ),
                "max_tokens": ("INT", {"default": 256, "min": 1, "max": 8192}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "random": ("BOOLEAN", {"default": False, "label_on": "Seed", "label_off": "Temperature"}),
            },
            "hidden": {
                "model": ("STRING", {"default": ""}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, engine, base_ip, port, profile):
        node = cls()
        if engine != node.engine or base_ip != node.base_ip or port != node.port or node.selected_model != node.get_models(engine, base_ip, port) or profile != node.profiles[node.selected_model]:
            node.engine = engine
            node.base_ip = base_ip
            node.port = port
            node.selected_model = node.get_models(engine, base_ip, port)
            node.profiles[node.selected_model] = profile
            return True
        return False
    
    def __init__(self):
        self.base_ip = "localhost" 
        self.port = "11434"     
        self.engine = "ollama" 
        self.selected_model = ""
        self.comfy_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.presets_dir = os.path.join(os.path.dirname(__file__), "presets")
        self.profiles_file = os.path.join(self.presets_dir, "profiles.json")
        self.profiles = self.load_presets(self.profiles_file)



    def load_presets(self, file_path):
        with open(file_path, 'r') as f:
            presets = json.load(f)
        return presets
   
    def get_api_key(self, api_key_name, engine):
        if engine != "ollama":  
            api_key = os.getenv(api_key_name)
            if api_key:
                return api_key
        else:
            print(f'you are using ollama as the engine, no api key is required')


    def get_models(self, engine, base_ip, port):
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
        elif engine == "anthropic":
            return ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
        elif engine == "openai":
            return ["gpt-4-0125-preview", "gpt-4-1106-preview", "gpt-4-vision-preview", "gpt-4-1106-vision-preview", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106"]
        else:
            print(f"Unsupported engine - {engine}")
            return []

    def tensor_to_image(self, tensor):
        # Ensure tensor is on CPU
        tensor = tensor.cpu()
        # Normalize tensor 0-255 and convert to byte
        image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
        # Create PIL image
        image = Image.fromarray(image_np, mode='RGB')
        return image


    def prepare_messages(self, image_prompt, profile):
        profile_selected = self.profiles.get(profile, "")
        
        if not image_prompt:
            system_message = textwrap.dedent("""
                When an image is provided, examine it closely and describe its contents, including the subject, setting, colors, composition, and any other relevant details.
                If the user asks a question about the image, provide a thoughtful and informative response based on your analysis.
                If no specific question is asked, provide a general description of the image.
            """)
        else:
            system_message = textwrap.dedent("""
                Please analyze the provided image and respond to the user's specific question or prompt.
                If no question is asked, provide a general description of the image, including the subject, setting, colors, composition, and any other relevant details.
            """)
        
        system_message = f"{profile_selected}\n{system_message}"
        
        user_message = image_prompt if image_prompt else "Please provide a general description of the image."
        
        return system_message, user_message

    
    def describe_picture(self, image, engine, selected_model, base_ip, port, image_prompt, profile, temperature, max_tokens, seed, random):
        
        if image is not None:
            # Check the type of the 'image' object
            if isinstance(image, torch.Tensor):
                # Convert the tensor to a PIL image
                pil_image = self.tensor_to_image(image)
            elif isinstance(image, Image.Image):
                pil_image = image
            elif isinstance(image, str) and os.path.isfile(image):
                pil_image = Image.open(image)
            else:
                print(f"Invalid image type: {type(image)}. Expected torch.Tensor, PIL.Image, or file path.")
                return "Invalid image type"

            # Convert the PIL image to base64
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        else:
            base64_image = None

        available_models = self.get_models(engine, base_ip, port)
        if available_models is None or selected_model not in available_models:
            error_message = f"Invalid model selected: {selected_model} for engine {engine}. Available models: {available_models}"
            print(error_message)
            raise ValueError(error_message)

        system_message, user_message = self.prepare_messages(image_prompt, profile)

        try:
            generated_text = self.send_request(engine, selected_model, base_ip, port, base64_image, system_message, user_message, temperature, max_tokens, seed, random)
            description = f"{generated_text}".strip()
            return  image_prompt, description
        except Exception as e:
            print(f"Exception occurred: {e}")
            return "Exception occurred while processing image."


    def send_request(self, engine, selected_model, base_ip, port, base64_image, system_message, user_message, temperature, max_tokens, seed, random):
        if base64_image is None:
            if engine == "anthropic":
                anthropic_api_key = self.get_api_key("ANTHROPIC_API_KEY", engine)
                anthropic_headers = {
                    "x-api-key": anthropic_api_key,
                    "anthropic-version": "2023-06-01",  
                    "Content-Type": "application/json"
                }
                if random == True:
                    data = {
                        "model": selected_model,
                        "system": system_message,
                        "messages": [
                            {
                                "role": "user",
                                "content": user_message
                            }
                        ],
                        "temperature": temperature,
                        "seed": seed,
                        "max_tokens": max_tokens
                    }
                else:
                    data = {
                        "model": selected_model,
                        "system": system_message,
                        "messages": [
                            {
                                "role": "user",
                                "content": user_message
                            }
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                
                api_url = 'https://api.anthropic.com/v1/messages'

                response = requests.post(api_url, headers=anthropic_headers, json=data)
                if response.status_code == 200:
                    response_data = response.json()
                    messages = response_data.get('content', [])
                    generated_text = ''.join([msg.get('text', '') for msg in messages if msg.get('type') == 'text'])
                    return generated_text
                else:
                    print(f"Error: Request failed with status code {response.status_code}, Response: {response.text}")
                    return "Failed to fetch response from Anthropic."
                    
            elif engine == "openai":
                openai_api_key = self.get_api_key("OPENAI_API_KEY", engine)
                openai_headers = {
                    "Authorization": f"Bearer {openai_api_key}",
                    "Content-Type": "application/json"
                }
                if random == True:
                    data = {
                        "model": selected_model,
                        "messages": [
                            {
                                "role": "system",
                                "content": system_message
                            },
                            {
                                "role": "user",
                                "content": user_message
                            }
                        ],
                        "temperature": temperature,
                        "seed": seed,
                        "max_tokens": max_tokens
                    }
                else:
                    data = {
                        "model": selected_model,
                        "messages": [
                            {
                                "role": "system",
                                "content": system_message
                            },
                            {
                                "role": "user",
                                "content": user_message
                            }
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }

                api_url = 'https://api.openai.com/v1/chat/completions'
                response = requests.post(api_url, headers=openai_headers, json=data)
                if response.status_code == 200:
                    response_data = response.json()
                    print("Debug Response:", response_data)
                    choices = response_data.get('choices', [])
                    if choices:
                        choice = choices[0]
                        message = choice.get('message', {})
                        generated_text = message.get('content', '')
                        return generated_text
                    else:
                        print("No valid choices in the response.")
                        print("Full response:", response.text)
                        return "No valid response generated."
                else:
                    print(f"Failed to fetch response, status code: {response.status_code}")
                    print("Full response:", response.text)
                    return "Failed to fetch response from OpenAI."

            else:
                api_url = f'http://{base_ip}:{port}/api/generate'
                if random == True:
                    data = {
                        "model": selected_model,
                        "system": system_message,
                        "prompt": user_message,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "seed": seed
                        }
                    }
                else:
                    data = {
                        "model": selected_model,
                        "system": system_message,
                        "prompt": user_message,
                        "stream": False,
                        "options": {
                            "temperature": temperature
                        }
                    }
                ollama_headers = {"Content-Type": "application/json"}
                response = requests.post(api_url, headers=ollama_headers, json=data)
                if response.status_code == 200:
                    response_data = response.json()
                    prompt_response = response_data.get('response', 'No response text found')
                    
                    # Ensure there is a response to construct the full description
                    if prompt_response != 'No response text found':
                        return prompt_response
                    else:
                        return "No valid response generated."
                else:
                    print(f"Failed to fetch response, status code: {response.status_code}")
                    return "Failed to fetch response from Ollama."
        
        else:
                
            if engine == "anthropic":
                anthropic_api_key = self.get_api_key(
                    "ANTHROPIC_API_KEY", engine)
                anthropic_headers = {
                    "x-api-key": anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                }
                if random == True:
                    data = {
                        "model": selected_model,
                        "system": system_message,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                                "media_type": "image/png",
                                                "data": base64_image
                                        }
                                    },
                                    {"type": "text", "text": user_message}
                                ]
                            }
                        ],
                        "temperature": temperature,
                        "seed": seed,
                        "max_tokens": max_tokens
                    }
                else:
                    data = {
                        "model": selected_model,
                        "system": system_message,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                                "media_type": "image/png",
                                                "data": base64_image
                                        }
                                    },
                                    {"type": "text", "text": user_message}
                                ]
                            }
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                    
                    api_url = 'https://api.anthropic.com/v1/messages'

                    response = requests.post(api_url, headers=anthropic_headers, json=data)
                    if response.status_code == 200:
                        response_data = response.json()
                        messages = response_data.get('content', [])
                        generated_text = ''.join([msg.get('text', '') for msg in messages if msg.get('type') == 'text'])
                        return generated_text
                    else:
                        print(f"Error: Request failed with status code {response.status_code}, Response: {response.text}")
                        return "Failed to fetch response from Anthropic."
                        
            elif engine == "openai":
                openai_api_key = self.get_api_key("OPENAI_API_KEY", engine)
                openai_headers = {
                    "Authorization": f"Bearer {openai_api_key}",
                    "Content-Type": "application/json"
                }
                if random == True:
                    data = {
                        "model": selected_model,
                        "messages": [
                            {
                                "role": "system",
                                "content": system_message
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": user_message
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": f"data:image/png;base64,{base64_image}"
                                    }
                                ]
                            }
                        ],
                        "temperature": temperature,
                        "seed": seed,
                        "max_tokens": max_tokens
                    }
                else:
                    data = {
                        "model": selected_model,
                        "messages": [
                            {
                                "role": "system",
                                "content": system_message
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": user_message
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": f"data:image/png;base64,{base64_image}"
                                    }
                                ]
                            }
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }

                    api_url = 'https://api.openai.com/v1/chat/completions'
                    response = requests.post(api_url, headers=openai_headers, json=data)
                    if response.status_code == 200:
                        response_data = response.json()
                        print("Debug Response:", response_data)
                        choices = response_data.get('choices', [])
                        if choices:
                            choice = choices[0]
                            message = choice.get('message', {})
                            generated_text = message.get('content', '')
                            return generated_text
                        else:
                            print("No valid choices in the response.")
                            print("Full response:", response.text)
                            return "No valid response generated for the image."
                    else:
                        print(f"Failed to fetch response, status code: {response.status_code}")
                        print("Full response:", response.text)
                        return "Failed to fetch response from OpenAI."

            else:
                api_url = f'http://{base_ip}:{port}/api/generate'
                if random == True:
                    data = {
                        "model": selected_model,
                        "system": system_message,
                        "prompt": user_message,
                        "stream": False,
                        "images": [base64_image],
                        "options": {
                            "temperature": temperature,
                            "seed": seed
                        }
                    }
                else:
                    data = {
                        "model": selected_model,
                        "system": system_message,
                        "prompt": user_message,
                        "stream": False,
                        "images": [base64_image],
                        "options": {
                            "temperature": temperature
                        }
                    }
                    ollama_headers = {"Content-Type": "application/json"}
                    response = requests.post(api_url, headers=ollama_headers, json=data)
                    if response.status_code == 200:
                        response_data = response.json()
                        prompt_response = response_data.get('response', 'No response text found')
                        
                        # Ensure there is a response to construct the full description
                        if prompt_response != 'No response text found':
                            return prompt_response
                        else:
                            return "No valid response generated for the image."
                    else:
                        print(f"Failed to fetch response, status code: {response.status_code}")
                        return "Failed to fetch response from Ollama."
            


NODE_CLASS_MAPPINGS = {"IF_ChatPrompt": IFChatPrompt}
NODE_DISPLAY_NAME_MAPPINGS = {"IF_ChatPrompt": "IF Chat Prompt👨‍💻"}
