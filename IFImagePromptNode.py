import requests
import base64
import textwrap
import io
import os
from io import BytesIO
from PIL import Image
import torch
import tempfile
import uuid
from torchvision.transforms.functional import to_pil_image
import folder_paths  

class IFImagePrompt:
    base_ip = "127.0.0.1"
    ollama_port = "11434"

    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.neg_prompts = self.load_presets(os.path.join(self.script_dir, "negfiles"))
        self.embellish_prompts = self.load_presets(os.path.join(self.script_dir, "embellishfiles"))
        self.style_prompts = self.load_presets(os.path.join(self.script_dir, "stylefiles"))

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
    def get_vision_models(cls):
        api_url = f'http://{cls.base_ip}:{cls.ollama_port}/api/tags'
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            vision_model = [model['name'] for model in response.json()['models']]
        except Exception as e:
            print(f"Failed to fetch models from Ollama: {e}")
            vision_model = []
        return vision_model

    def describe_picture(self, image, select_vision_model, image_prompt=None, embellish_prompt=None, style_prompt=None, neg_prompt=None,
                        temperature=0.7, seed=0, top_k=40, repeat_penalty=1.1, num_ctx=2048):
        embellish_content = next((content for name, content in self.embellish_prompts if name == embellish_prompt), "")
        style_content = next((content for name, content in self.style_prompts if name == style_prompt), "")
        neg_content = next((content for name, content in self.neg_prompts if name == neg_prompt), "")

        temperature = temperature or 0.7
        seed = seed or 0
        top_k = top_k or 40
        repeat_penalty = repeat_penalty or 1.1
        num_ctx = num_ctx or 2048

        image_path = folder_paths.get_annotated_filepath(image)
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        system_message, user_message = self._prepare_messages(image_prompt)

        api_url = f'http://{self.base_ip}:{self.ollama_port}/api/generate'
        data = {
            "model": select_vision_model,
            "system": system_message,
            "prompt": user_message,
            "stream": False,
            "images": [base64_image],
            "options": {
                "temperature": temperature,
                "seed": seed,
                "top_k": top_k,
                "repeat_penalty": repeat_penalty,
                "num_ctx": num_ctx,
            },
        }
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(api_url, headers=headers, json=data)
            if response.status_code == 200:
                response_data = response.json()
                prompt_response = response_data.get('response', 'No response text found')
                print(f"Caption for image {os.path.basename(image_path)}: {prompt_response}")
                
                # Ensure there is a response to construct the full description
                if prompt_response != 'No response text found':
                    description = f"{embellish_content} {prompt_response} {style_content}".strip()
                    return image_prompt, description, neg_content
                else:
                    return "No valid response generated for the image.", neg_content
            else:
                print(f"Failed to fetch response, status code: {response.status_code}")
                return image_prompt, "Failed to fetch response from Ollama.", neg_content
        except Exception as e:
            print(f"Exception occurred: {e}")
            return image_prompt, "Exception occurred while processing image.", neg_content


    def _prepare_messages(self, image_prompt=None):
        system_message = textwrap.dedent("""\
                    Act as a visual prompt maker with the following guidelines:
                    - Describe the image in vivid detail.
                    - Break keywords by commas.
                    - Provide high-quality, non-verbose, coherent, concise, and not superfluous descriptions.
                    - Focus solely on the visual elements of the picture; avoid art commentaries or intentions.
                    - Construct the prompt by describing framing, subjects, scene elements, background, aesthetics.
                    - Limit yourself up to 7 keywords per component  
                    - Be varied and creative.
                    - Always reply on the same line, use around 100 words long. 
                    - Do not enumerate or enunciate components.
                    - Do not include any additional information in the response.                                                       
                    The following is an illustrative example for you to see how to construct a prompt your prompts should follow this format but always coherent to the subject worldbuilding or setting and consider the elements relationship:
                    'Epic, Cover Art, Full body shot, dynamic angle, A Demon Hunter, standing, lone figure, glow eyes, deep purple light, cybernetic exoskeleton, sleek, metallic, glowing blue accents, energy weapons. Fighting Demon, grotesque creature, twisted metal, glowing red eyes, sharp claws, Cyber City, towering structures, shrouded haze, shimmering energy. Ciberpunk, dramatic lighthing, highly detailed. ' 
                    Make a visual prompt for the following Image:
                    """) if not image_prompt else "Please analyze the image and respond to the user's question."
        user_message = image_prompt if image_prompt else textwrap.dedent("""\
                    Act as a visual prompt maker with the following guidelines:
                    - Describe the image in vivid detail.
                    - Break keywords by commas.
                    - Provide high-quality, non-verbose, coherent, concise, and not superfluous descriptions.
                    - Focus solely on the visual elements of the picture; avoid art commentaries or intentions.
                    - Construct the prompt by describing framing, subjects, scene elements, background, aesthetics.
                    - Limit yourself up to 7 keywords per component  
                    - Be varied and creative.
                    - Always reply on the same line, use around 100 words long. 
                    - Do not enumerate or enunciate components.
                    - Do not include any additional information in the response.                                                       
                    The following is an illustartive example for you to see how to construct a prompt your prompts should follow this format but always coherent to the subject worldbuilding or setting and consider the elements relationship:
                    'Epic, Cover Art, Full body shot, dynamic angle, A Demon Hunter, standing, lone figure, glow eyes, deep purple light, cybernetic exoskeleton, sleek, metallic, glowing blue accents, energy weapons. Fighting Demon, grotesque creature, twisted metal, glowing red eyes, sharp claws, Cyber City, towering structures, shrouded haze, shimmering energy. Ciberpunk, dramatic lighthing, highly detailed. ' 
                    Make a visual prompt for the following Image:
                    """)
        return system_message, user_message
        


    @classmethod
    def INPUT_TYPES(cls):
        node = cls()
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image_prompt": ("STRING", {"multiline": True, "default": ""}),
                "image": (sorted(files), {"image_upload": True}),
                "select_vision_model": (cls.get_vision_models(), {}),
                "embellish_prompt": ([name for name, _ in node.embellish_prompts], {}),
                "style_prompt": ([name for name, _ in node.style_prompts], {}),
                "neg_prompt": ([name for name, _ in node.neg_prompts], {}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967295}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 2.0, "step": 0.1}),
                "num_ctx": ("INT", {"default": 2048, "min": 64, "max": 8192}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("Question", "Response", "Negative",)
    FUNCTION = "describe_picture"
    OUTPUT_NODE = True
    CATEGORY = "ImpactFrames💥🎞️"



NODE_CLASS_MAPPINGS = {"IF_ImagePrompt": IFImagePrompt}
NODE_DISPLAY_NAME_MAPPINGS = {"IF_ImagePrompt": "IF Image to Prompt🖼️"}
