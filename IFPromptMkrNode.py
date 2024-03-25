import requests
import os
import textwrap
import anthropic
import openai
from server import PromptServer
from aiohttp import web

class IFPrompt2Prompt:

    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.neg_prompts = self.load_presets(os.path.join(self.script_dir, "negfiles"))
        self.embellish_prompts = self.load_presets(os.path.join(self.script_dir, "embellishfiles"))
        self.style_prompts = self.load_presets(os.path.join(self.script_dir, "stylefiles"))
        self.base_ip = "127.0.0.1"
        self.ollama_port = "11434"
        self.anthropic_api_key = self.get_api_key("ANTHROPIC_API_KEY")
        self.openai_api_key = self.get_api_key("OPENAI_API_KEY")

        self.prime_directive = textwrap.dedent("""\
            Act as a prompt maker with the following guidelines:
            - Break keywords by commas.
            - Provide high-quality, non-verbose, coherent, brief, concise, and not superfluous prompts.
            - Focus solely on the visual elements of the picture; avoid art commentaries or intentions.
            - Construct the prompt with the component format:
            1. Start with the subject and keyword description.
            2. Follow with scene keyword description.
            3. Finish with background and keyword description.
            - Limit yourself to no more than 7 keywords per component  
            - Include all the keywords from the user's request verbatim as the main subject of the response.
            - Be varied and creative.
            - Always reply on the same line and no more than 100 words long. 
            - Do not enumerate or enunciate components.
            - Do not include any additional information in the response.                                                       
            The following is an illustrative example for you to see how to construct a prompt your prompts should follow this format but always coherent to the subject worldbuilding or setting and consider the elements relationship.
            Example:
            Subject: Demon Hunter, Cyber City.
            prompt: A Demon Hunter, standing, lone figure, glow eyes, deep purple light, cybernetic exoskeleton, sleek, metallic, glowing blue accents, energy weapons. Fighting Demon, grotesque creature, twisted metal, glowing red eyes, sharp claws, Cyber City, towering structures, shrouded haze, shimmering energy.                             
            Make a prompt for the following Subject:
            """)

    def load_presets(self, dir_path):
        presets = []
        for filename in os.listdir(dir_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(dir_path, filename)
                with open(file_path, 'r') as file:
                    content = file.read().strip()
                    presets.append((os.path.splitext(filename)[0], content))
        return presets

    def get_api_key(self, api_key_name):
        api_key = os.getenv(api_key_name)
        if api_key:
            print(f"API key found for {api_key_name}")
            return api_key
        print(f"Error: {api_key_name} is required")
        return ""

    def get_models(self, engine, base_ip, ollama_port):
        if engine == "ollama":
            api_url = f'http://{base_ip}:{ollama_port}/api/tags'
            try:
                response = requests.get(api_url)
                response.raise_for_status()
                return [model['name'] for model in response.json().get('models', [])]
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

    def sample(self, input_prompt, engine, selected_model, embellish_prompt, style_prompt, neg_prompt, base_ip, ollama_port, temperature, max_tokens):
        embellish_content = next((content for name, content in self.embellish_prompts if name == embellish_prompt), "")
        style_content = next((content for name, content in self.style_prompts if name == style_prompt), "")
        neg_content = next((content for name, content in self.neg_prompts if name == neg_prompt), "")

        available_models = self.get_models(engine, base_ip, ollama_port)
        if available_models is None or selected_model not in available_models:
            error_message = f"Invalid model selected: {selected_model} for engine {engine}. Available models: {available_models}"
            print(error_message)
            raise ValueError(error_message)
            
        if engine == "anthropic":
            data = {
                'model': selected_model,
                'system': self.prime_directive, 
                'messages': [
                    {"role": "user", "content": input_prompt}  
                ],
                'temperature': temperature,
                'max_tokens': max_tokens      
            }
        elif engine == "openai":
            data = {
                'model': selected_model, 
                'messages': [
                    {"role": "system", "content": self.prime_directive},
                    {"role": "user", "content": input_prompt}
                ],
                'temperature': temperature,
                'max_tokens': max_tokens  
            }
        else:
            data = {
                'model': selected_model, 
                'messages': [
                    {"role": "system", "content": self.prime_directive},
                    {"role": "user", "content": input_prompt}
                ],
                'temperature': temperature,
                'max_tokens': max_tokens
            }

        generated_text = self.send_request(data, headers={"Content-Type": "application/json"}, engine=engine)
 
        if generated_text:
            combined_prompt = f"{embellish_content} {generated_text} {style_content}"
            return input_prompt, combined_prompt, neg_content
        else:
            return None, None, None

    def send_request(self, data, headers, engine):
        if engine == "ollama":
            base_url = f'http://{self.base_ip}:{self.ollama_port}/v1/chat/completions'
            response = requests.post(base_url, headers=headers, json=data)
            if response.status_code == 200:
                return response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
            else:
                print(f"Error: Request failed with status code {response.status_code}")
                return None
        elif engine == "anthropic":
            try:
                base_url = 'https://api.anthropic.com/v1/messages'
                anthropic_headers = {
                    "x-api-key": self.anthropic_api_key,
                    "anthropic-version": "2023-06-01",  
                    "Content-Type": "application/json"
                }
                response = requests.post(base_url, headers=anthropic_headers, json=data)
                if response.status_code == 200:
                    messages = response.json().get('content', [])
                    generated_text = ''.join([msg.get('text', '') for msg in messages if msg.get('type') == 'text'])
                    return generated_text
                else:
                    print(f"Error: Request failed with status code {response.status_code}, Response: {response.text}")
                    return None
            except Exception as e:
                print(f"Error: Anthropic request failed - {e}")
                return None
        elif engine == "openai":
            try:
                base_url = 'https://api.openai.com/v1/chat/completions'
                openai_headers = {
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                }
                response = requests.post(base_url, headers=openai_headers, json=data)
                if response.status_code == 200:
                    response_data = response.json()
                    print("Debug Response:", response_data)  
                    choices = response_data.get('choices', [])
                    if choices:
                        choice = choices[0]
                        messages = choice.get('message', {'content': ''})  
                        generated_text = messages.get('content', '') 
                        return generated_text
                    else:
                        print("No choices found in response")
                        return None
                else:
                    print(f"Error: Request failed with status code {response.status_code}, Response: {response.text}")
                    return None
            except Exception as e:
                print(f"Error: OpenAI request failed - {e}")
                return None, None, None

    @classmethod
    def INPUT_TYPES(cls):
        node = cls()
        default_engine = "ollama"
        return {
            "required": {
                "input_prompt": ("STRING", {"multiline": True, "default": "Ancient mega-structure, small lone figure in the foreground"}),
                "engine": (["ollama", "openai", "anthropic"], {"default": default_engine}),
                "selected_model": (node.get_models(default_engine, node.base_ip, node.ollama_port) + node.get_models("anthropic", node.base_ip, node.ollama_port) + node.get_models("openai", node.base_ip, node.ollama_port), {}),
                "embellish_prompt": ([name for name, _ in node.embellish_prompts], {}),
                "style_prompt": ([name for name, _ in node.style_prompts], {}),
                "neg_prompt": ([name for name, _ in node.neg_prompts], {}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
            },
            "optional": {
                "max_tokens": ("INT", {"default": 160, "min": 1, "max": 2048}),
                "base_ip": ("STRING", {"default": node.base_ip}),
                "ollama_port": ("STRING", {"default": node.ollama_port}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("Question", "Response", "Negative",)
    FUNCTION = "sample"
    OUTPUT_NODE = False
    CATEGORY = "ImpactFrames💥🎞️"


    @PromptServer.instance.routes.post("/IF_PromptMkr/get_models")
    async def get_models_endpoint(request):
        data = await request.json()
        engine = data.get("engine")
        base_ip = data.get("base_ip")
        ollama_port = data.get("ollama_port")
        
        node = IFPrompt2Prompt()
        models = node.get_models(engine, base_ip, ollama_port)
        
        return web.json_response(models)

NODE_CLASS_MAPPINGS = {"IF_PromptMkr": IFPrompt2Prompt}
NODE_DISPLAY_NAME_MAPPINGS = {"IF_PromptMkr": "IF Prompt to Prompt💬"}
