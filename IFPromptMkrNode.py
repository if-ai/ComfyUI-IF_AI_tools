# You need to install ollama and have it running on your machine for this to work properly.
import requests
import os
import textwrap


class IFPromptMkrNode:
    base_ip = "127.0.0.1"
    ollama_port = "11434"

    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.neg_prompts = self.load_presets(os.path.join(self.script_dir, "negfiles"))
        self.embellish_prompts = self.load_presets(os.path.join(self.script_dir, "embellishfiles"))
        self.style_prompts = self.load_presets(os.path.join(self.script_dir, "stylefiles"))

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
            The followin is an illustartive example for you to see how to construct a prompt your prompts should follow this format but always coherent to the subject worldbuilding or setting and cosider the elemnts relationship.
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
    
    @classmethod
    def get_text_models(cls):
        api_url = f'http://{cls.base_ip}:{cls.ollama_port}/api/tags'
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            text_models = [model['name'] for model in response.json()['models']]
        except Exception as e:
            print(f"Failed to fetch models from Ollama: {e}")
            text_models = []
        return text_models

    def send_request(self, data, headers):
        base_url = f'http://{self.base_ip}:{self.ollama_port}/v1/chat/completions'
        response = requests.post(base_url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
        else:
            print(f"Error: Request failed with status code {response.status_code}")
            return None

    def sample(self, input_prompt, select_text_model, embellish_prompt, style_prompt, neg_prompt):
        # Look up the content by name
        embellish_content = next((content for name, content in self.embellish_prompts if name == embellish_prompt), "")
        style_content = next((content for name, content in self.style_prompts if name == style_prompt), "")
        neg_content = next((content for name, content in self.neg_prompts if name == neg_prompt), "")
        
        data = {
            'model': select_text_model, 
            'messages': [
                {"role": "system", "content": self.prime_directive},
                {"role": "user", "content": input_prompt}
            ],
        }
        
        generated_text = self.send_request(data, headers={"Content-Type": "application/json"})
        
        if generated_text:
            # Combine using the content, not the name
            combined_prompt = f"{embellish_content} {generated_text} {style_content}"
            return input_prompt, combined_prompt, neg_content
        else:
            return None, None, None

    @classmethod
    def INPUT_TYPES(cls):
        node = cls()  # Create an instance of the class
        return {
            "required": {
                "input_prompt": ("STRING", {"multiline": True, "default": "lone figure, ancient Megastructure"}),
                "select_text_model": (cls.get_text_models(), {}),
                "embellish_prompt": ([name for name, _ in node.embellish_prompts], {}),
                "style_prompt": ([name for name, _ in node.style_prompts], {}),
                "neg_prompt": ([name for name, _ in node.neg_prompts], {}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("Question", "Response", "Negative",)
    FUNCTION = "sample"
    OUTPUT_NODE = False
    CATEGORY = "ImpactFrames💥🎞️"


NODE_CLASS_MAPPINGS = {"IFPromptMkrNode": IFPromptMkrNode}
NODE_DISPLAY_NAME_MAPPINGS = {"IFPromptMkrNode": "IF Prompt to Prompt💬"}