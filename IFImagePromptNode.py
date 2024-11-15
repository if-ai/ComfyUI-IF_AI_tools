# IFImagePromptNode.py
import os
import sys
import json
import torch
import asyncio
import requests
from PIL import Image
from io import BytesIO
from typing import List, Dict, Any, Optional, Union, Tuple
import folder_paths
from .omost import omost_function
from .send_request import send_request
from .utils import (
    get_api_key,
    get_models,
    process_images_for_comfy,
    process_mask,
    clean_text,
    load_placeholder_image,
    validate_models,
)

# Add ComfyUI directory to path
comfy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, comfy_path)

# Set up logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.post("/IF_ImagePrompt/get_llm_models")
    async def get_llm_models_endpoint(request):
        try:
            data = await request.json()
            llm_provider = data.get("llm_provider")
            engine = llm_provider
            base_ip = data.get("base_ip")
            port = data.get("port")
            external_api_key = data.get("external_api_key")
        
            if external_api_key:
                api_key = external_api_key
            else:
                api_key_name = f"{llm_provider.upper()}_API_KEY"
                try:
                    api_key = get_api_key(api_key_name, engine)
                except ValueError:
                    api_key = None

            node = IFImagePrompt()
            models = node.get_models(engine, base_ip, port, api_key)
            return web.json_response(models)
        
        except Exception as e:
            print(f"Error in get_llm_models_endpoint: {str(e)}")
            return web.json_response([], status=500)

    @PromptServer.instance.routes.post("/IF_ImagePrompt/add_routes")
    async def add_routes_endpoint(request):
        return web.json_response({"status": "success"})

except AttributeError:
    print("PromptServer.instance not available. Skipping route decoration for IF_ImagePrompt.")

class IFImagePrompt:
    def __init__(self):
        self.strategies = "normal"
        # Initialize paths and load presets
        self.base_path = folder_paths.base_path
        self.presets_dir = os.path.join(self.base_path, "custom_nodes", "ComfyUI-IF_AI_tools", "IF_AI", "presets")

        # Load preset configurations
        self.profiles = self.load_presets(os.path.join(self.presets_dir, "profiles.json"))
        self.neg_prompts = self.load_presets(os.path.join(self.presets_dir, "neg_prompts.json"))
        self.embellish_prompts = self.load_presets(os.path.join(self.presets_dir, "embellishments.json"))
        self.style_prompts = self.load_presets(os.path.join(self.presets_dir, "style_prompts.json"))
        self.stop_strings = self.load_presets(os.path.join(self.presets_dir, "stop_strings.json"))

        # Initialize placeholder image path
        self.placeholder_image_path = os.path.join(folder_paths.base_path, "custom_nodes", "ComfyUI-IF_AI_tools", "IF_AI", "placeholder.png")

        # Default values

        self.base_ip = "localhost"
        self.port = "11434"
        self.engine = "xai"
        self.selected_model = ""
        self.profile = "IF_PromptMKR_IMG"
        self.messages = []
        self.keep_alive = False
        self.seed = 94687328150
        self.history_steps = 10
        self.external_api_key = ""
        self.preset = "Default"
        self.precision = "fp16"
        self.attention = "sdpa"
        self.Omni = None
        self.mask = None
        self.aspect_ratio = "1:1"
        self.keep_alive = False
        self.clear_history = False
        self.random = False
        self.max_tokens = 2048
        self.temperature = 0.7
        self.top_k = 40
        self.top_p = 0.9
        self.repeat_penalty = 1.1
        self.stop = None
        self.batch_count = 4

    @classmethod
    def INPUT_TYPES(cls):
        node = cls() 
        return {
            "required": {
                "images": ("IMAGE", {"list": True}),  # Primary image input
                "llm_provider": (["xai","llamacpp", "ollama", "kobold", "lmstudio", "textgen", "groq", "gemini", "openai", "anthropic", "mistral", "transformers"], {}),
                "llm_model": ((), {}),
                "base_ip": ("STRING", {"default": "localhost"}),
                "port": ("STRING", {"default": "11434"}),
                "user_prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "strategy": (["normal", "omost", "create", "edit", "variations"], {"default": "normal"}),
                "mask": ("MASK", {}),
                "prime_directives": ("STRING", {"forceInput": True, "tooltip": "The system prompt for the LLM."}),
                "profiles": (["None"] + list(cls().profiles.keys()), {"default": "None", "tooltip": "The pre-defined system_prompt from the json profile file on the presets folder you can edit or make your own will be listed here."}),
                "embellish_prompt": (list(cls().embellish_prompts.keys()), {"tooltip": "The pre-defined embellishment from the json embellishments file on the presets folder you can edit or make your own will be listed here."}),
                "style_prompt": (list(cls().style_prompts.keys()), {"tooltip": "The pre-defined style from the json style_prompts file on the presets folder you can edit or make your own will be listed here."}),
                "neg_prompt": (list(cls().neg_prompts.keys()), {"tooltip": "The pre-defined negative prompt from the json neg_prompts file on the presets folder you can edit or make your own will be listed here."}),
                "stop_string": (list(cls().stop_strings.keys()), {"tooltip": "Specifies a string at which text generation should stop."}),
                "max_tokens": ("INT", {"default": 2048, "min": 1, "max": 8192, "tooltip": "Maximum number of tokens to generate in the response."}),
                "random": ("BOOLEAN", {"default": False, "label_on": "Seed", "label_off": "Temperature", "tooltip": "Toggles between using a fixed seed or temperature-based randomness."}),
                "seed": ("INT", {"default": 0, "tooltip": "Random seed for reproducible outputs."}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "tooltip": "Controls randomness in output generation. Higher values increase creativity but may reduce coherence."}),
                "top_k": ("INT", {"default": 40, "tooltip": "Limits the next token selection to the K most likely tokens."}),
                "top_p": ("FLOAT", {"default": 0.9, "tooltip": "Cumulative probability cutoff for token selection."}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "tooltip": "Penalizes repetition in generated text."}),
                "keep_alive": ("BOOLEAN", {"default": False, "label_on": "Keeps Model on Memory", "label_off": "Unloads Model from Memory", "tooltip": "Determines whether to keep the model loaded in memory between calls."}),
                "clear_history": ("BOOLEAN", {"default": False, "label_on": "Clear History", "label_off": "Keep History", "tooltip": "Determines whether to clear the history between calls."}),
                "history_steps": ("INT", {"default": 10, "tooltip": "Number of steps to keep in history."}),
                "aspect_ratio": (["1:1", "16:9", "4:5", "3:4", "5:4", "9:16"], {"default": "1:1", "tooltip": "Aspect ratio for the generated images."}),
                "batch_count": ("INT", {"default": 4, "tooltip": "Number of images to generate. only for create, edit and variations strategies."}),
                "external_api_key": ("STRING", {"default": "", "tooltip": "If this is not empty, it will be used instead of the API key from the .env file. Make sure it is empty to use the .env file."}),
                "precision": (["fp16", "fp32", "bf16"], {"tooltip": "Select preccision on Transformer models."}),
                "attention": (["sdpa", "flash_attention_2", "xformers"], {"tooltip": "Select attention mechanism on Transformer models."}),
                "Omni": ("OMNI", {"default": None, "tooltip": "Additional input for the selected tool."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "OMNI", "IMAGE", "MASK")
    RETURN_NAMES = ("question", "response", "negative", "omni", "generated_images", "mask")

    FUNCTION = "process_image_wrapper"
    OUTPUT_NODE = True
    CATEGORY = "ImpactFrames💥🎞️"

    def get_models(self, engine, base_ip, port, api_key=None):
        return get_models(engine, base_ip, port, api_key)

    def load_presets(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading presets from {file_path}: {e}")
            return {}

    def validate_outputs(self, outputs):
        """Helper to validate output types match expectations"""
        if len(outputs) != len(self.RETURN_TYPES):
            raise ValueError(
                f"Expected {len(self.RETURN_TYPES)} outputs, got {len(outputs)}"
            )

        for i, (output, expected_type) in enumerate(zip(outputs, self.RETURN_TYPES)):
            if output is None and expected_type in ["IMAGE", "MASK"]:
                raise ValueError(
                    f"Output {i} ({self.RETURN_NAMES[i]}) cannot be None for type {expected_type}"
                )

    async def generate_negative_prompts(
        self,
        prompt: str,
        llm_provider: str,
        llm_model: str,
        base_ip: str,
        port: str,
        config: dict,
        messages: list = None
    ) -> List[str]:
        """
        Generate negative prompts for the given input prompt.
        
        Args:
            prompt: Input prompt text
            llm_provider: LLM provider name
            llm_model: Model name
            base_ip: API base IP
            port: API port
            config: Dict containing generation parameters like seed, temperature etc
            messages: Optional message history
            
        Returns:
            List of generated negative prompts
        """
        try:
            if not prompt:
                return []

            # Get system message for negative prompts
            neg_system_message = self.profiles.get("IF_NegativePromptEngineer", "")

            # Generate negative prompts
            neg_response = await send_request(
                llm_provider=llm_provider,
                base_ip=base_ip,
                port=port,
                images=None,
                llm_model=llm_model,
                system_message=neg_system_message,
                user_message=f"Generate negative prompts for:\n{prompt}",
                messages=messages or [],
                **config
            )

            if not neg_response:
                return []

            # Split into lines and clean up
            neg_lines = [line.strip() for line in neg_response.split('\n') if line.strip()]

            # Match number of prompts
            num_prompts = len(prompt.split('\n'))
            if len(neg_lines) < num_prompts:
                neg_lines.extend([neg_lines[-1] if neg_lines else ""] * (num_prompts - len(neg_lines)))

            return neg_lines[:num_prompts]

        except Exception as e:
            logger.error(f"Error generating negative prompts: {str(e)}")
            return ["Error generating negative prompt"] * num_prompts

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    async def process_image(
        self,
        llm_provider: str,
        llm_model: str,
        base_ip: str,
        port: str,
        user_prompt: str,
        strategy: str = "normal",
        images=None,
        prime_directives: Optional[str] = None,
        profiles: Optional[str] = None,
        embellish_prompt: Optional[str] = None,
        style_prompt: Optional[str] = None,
        neg_prompt: Optional[str] = None,
        stop_string: Optional[str] = None,
        max_tokens: int = 2048,
        seed: int = 0,
        random: bool = False,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        keep_alive: bool = False,
        clear_history: bool = False,
        history_steps: int = 10,
        external_api_key: str = "",
        precision: str = "fp16",
        attention: str = "sdpa",
        Omni: Optional[str] = None,
        aspect_ratio: str = "1:1",
        mask: Optional[torch.Tensor] = None,
        batch_count: int = 4,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        try:
            # Initialize variables at the start
            formatted_response = None
            generated_images = None
            generated_masks = None
            tool_output = None

            if external_api_key != "":
                llm_api_key = external_api_key
            else:
                llm_api_key = get_api_key(f"{llm_provider.upper()}_API_KEY", llm_provider)
            print(f"LLM API key: {llm_api_key[:5]}...")

            # Validate LLM model
            validate_models(llm_model, llm_provider, "LLM", base_ip, port, llm_api_key)

            # Handle history
            if clear_history:
                self.messages = []
            elif history_steps > 0:
                self.messages = self.messages[-history_steps:]

            messages = self.messages

            # Handle stop
            if stop_string is None or stop_string == "None":
                stop_content = None
            else:
                stop_content = self.stop_strings.get(stop_string, None)
            stop = stop_content

            if llm_provider not in ["ollama", "llamacpp", "vllm", "lmstudio", "gemeni"]:
                if llm_provider == "kobold":
                    stop = stop_content + \
                        ["\n\n\n\n\n"] if stop_content else ["\n\n\n\n\n"]
                elif llm_provider == "mistral":
                    stop = stop_content + \
                        ["\n\n"] if stop_content else ["\n\n"]
                else:
                    stop = stop_content if stop_content else None

            # Prepare embellishments and styles
            embellish_content = self.embellish_prompts.get(embellish_prompt, "").strip() if embellish_prompt else ""
            style_content = self.style_prompts.get(style_prompt, "").strip() if style_prompt else ""
            neg_content = self.neg_prompts.get(neg_prompt, "").strip() if neg_prompt else ""
            profile_content = self.profiles.get(profiles, "")

            # Prepare system prompt
            if prime_directives is not None:
                system_message_str = prime_directives
            else:
                system_message_str= json.dumps(profile_content)

            if strategy == "omost":
                system_prompt = self.profiles.get("IF_Omost")
                messages = []
                # Generate the text using LLM
                llm_response = await send_request(
                    llm_provider=llm_provider,
                    base_ip=base_ip,
                    port=port,
                    images=images,
                    llm_model=llm_model,
                    system_message=system_prompt,
                    user_message=user_prompt,
                    messages=messages,
                    seed=seed,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    random=random,
                    top_k=top_k,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    stop=stop,
                    keep_alive=keep_alive,
                    llm_api_key=llm_api_key,
                    tools=None,
                    tool_choice=None,
                    precision=precision, 
                    attention=attention,
                    aspect_ratio=aspect_ratio,
                    strategy="omost",
                    batch_count=batch_count,
                    mask=mask,
                    )

                # Pass the generated_text to omost_function
                tool_args = {
                    "name": "omost_tool",
                    "description": "Analyzes images composition and generates a Canvas representation.",
                    "system_prompt": system_prompt,
                    "input": user_prompt,
                    "llm_response": llm_response,
                    "function_call": None,
                    "omni_input": Omni
                }

                tool_result = await omost_function(tool_args)

                # Process the tool output
                if "error" in tool_result:
                    llm_response = f"Error: {tool_result['error']}"
                    tool_output = None
                else:
                    tool_output = tool_result.get("canvas_conditioning", "")
                    llm_response = f"{tool_output}"
                    cleaned_response = clean_text(llm_response)

                neg_content = self.neg_prompts.get(neg_prompt, "").strip() if neg_prompt else ""

                # Update message history if keeping alive
                if keep_alive and cleaned_response:
                    messages.append({"role": "user", "content": user_prompt})
                    messages.append({"role": "assistant", "content": cleaned_response})

                return {
                    "Question": user_prompt,
                    "Response": cleaned_response,
                    "Negative": neg_content,
                    "Tool_Output": tool_output,
                    "Retrieved_Image": None,
                    "Mask": None
                }
            elif strategy in ["create", "edit", "variations"]:
                resulting_images = await send_request(
                    llm_provider=llm_provider,
                    base_ip=base_ip,
                    port=port,
                    images=images,
                    llm_model=llm_model,
                    system_message=system_prompt,
                    user_message=user_prompt,
                    messages=messages,
                    seed=seed,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    random=random,
                    top_k=top_k,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    stop=stop,
                    keep_alive=keep_alive,
                    llm_api_key=llm_api_key,
                    tools=None,
                    tool_choice=None,
                    precision=precision, 
                    attention=attention,
                    aspect_ratio=aspect_ratio,
                    strategy=strategy,
                    batch_count=batch_count,
                    mask=mask,
                )
                if isinstance(resulting_images, dict) and "images" in resulting_images:
                    generated_images = resulting_images["images"]
                    generated_masks = None
                else:
                    generated_images = None
                    generated_masks = None

                try: 
                    if generated_images is not None:
                        if isinstance(generated_images, torch.Tensor):
                            # Ensure correct format (B, C, H, W)
                            image_tensor = generated_images.unsqueeze(0) if generated_images.dim() == 3 else generated_images

                            # Create matching batch masks
                            batch_size = image_tensor.shape[0]
                            height = image_tensor.shape[2]
                            width = image_tensor.shape[3]

                            # Create default masks
                            mask_tensor = torch.ones((batch_size, 1, height, width), 
                                                dtype=torch.float32,
                                                device=image_tensor.device)

                            if generated_masks is not None:
                                mask_tensor = process_mask(generated_masks, image_tensor)
                        else:
                            image_tensor, mask_tensor = process_images_for_comfy(generated_images, self.placeholder_image_path)
                            mask_tensor = process_mask(generated_masks, image_tensor) if generated_masks is not None else mask_tensor
                    else:
                        # No retrieved image - use original or placeholder
                        if images is not None and len(images) > 0:
                            image_tensor = images[0] if isinstance(images[0], torch.Tensor) else process_images_for_comfy(images, self.placeholder_image_path)[0]
                            mask_tensor = torch.ones_like(image_tensor[:1]) # Create mask with same spatial dimensions
                        else:
                            image_tensor, mask_tensor = load_placeholder_image(self.placeholder_image_path)

                    return {
                            "Question": user_prompt,
                            "Response": f"{strategy} image has been successfully generated.",
                            "Negative": neg_content,
                            "Tool_Output": None,
                            "Retrieved_Image": image_tensor,
                            "Mask": mask_tensor
                        }

                except Exception as e:
                    print(f"Error in process_image: {str(e)}")
                    image_tensor, mask_tensor = load_placeholder_image(self.placeholder_image_path)
                    return {
                        "Question": user_prompt,
                        "Response": f"Error: {str(e)}",
                        "Negative": "",
                        "Tool_Output": None,
                        "Retrieved_Image": image_tensor,
                        "Mask": mask_tensor
                    }
            elif strategy == "normal":
                try:
                    formatted_responses = []
                    final_prompts = []
                    final_negative_prompts = []
                    
                    # Handle images as they come from ComfyUI - no extra processing needed
                    current_images = images if images is not None else None
                    
                    # If mask provided, ensure it matches image dimensions
                    if mask is not None:
                        mask_tensor = process_mask(mask, current_images)
                    else:
                        # Create default mask if needed
                        if current_images is not None:
                            mask_tensor = torch.ones((current_images.shape[0], 1, current_images.shape[2], current_images.shape[3]), 
                                                dtype=torch.float32,
                                                device=current_images.device)
                        else:
                            _, mask_tensor = load_placeholder_image(self.placeholder_image_path)

                    # Iterate over batches
                    for batch_idx in range(batch_count):
                        try:
                            response = await send_request(
                                llm_provider=llm_provider,
                                base_ip=base_ip,
                                port=port,
                                images=current_images,  # Pass images directly
                                llm_model=llm_model,
                                system_message=system_message_str,
                                user_message=user_prompt,
                                messages=messages,
                                seed=seed + batch_idx if seed != 0 else seed,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                random=random,
                                top_k=top_k,
                                top_p=top_p,
                                repeat_penalty=repeat_penalty,
                                stop=stop,
                                keep_alive=keep_alive,
                                llm_api_key=llm_api_key,
                                precision=precision,
                                attention=attention,
                                aspect_ratio=aspect_ratio,
                                strategy="normal",
                                batch_count=1,
                                mask=mask_tensor,
                            )

                            if not response:
                                raise ValueError("No response received from LLM API")
                            
                            # Clean and process response 
                            cleaned_response = clean_text(response)
                            final_prompts.append(cleaned_response)
                            
                            # Handle negative prompts
                            if neg_prompt == "AI_Fill":
                                negative_prompt = await self.generate_negative_prompts(
                                    prompt=cleaned_response,
                                    llm_provider=llm_provider,
                                    llm_model=llm_model,
                                    base_ip=base_ip,
                                    port=port,
                                    config={
                                        "seed": seed + batch_idx if seed != 0 else seed,
                                        "temperature": temperature,
                                        "max_tokens": max_tokens,
                                        "random": random,
                                        "top_k": top_k,
                                        "top_p": top_p,
                                        "repeat_penalty": repeat_penalty
                                    },
                                    messages=messages
                                )
                                final_negative_prompts.append(negative_prompt[0] if negative_prompt else neg_content)
                            else:
                                final_negative_prompts.append(neg_content)
                                
                            formatted_responses.append(cleaned_response)
                            
                        except Exception as e:
                            logger.error(f"Error in batch {batch_idx}: {str(e)}")
                            formatted_responses.append(f"Error in batch {batch_idx}: {str(e)}")
                            final_negative_prompts.append(f"Error generating negative prompt for batch {batch_idx}")
                    
                    # Combine all responses
                    formatted_response = "\n".join(final_prompts)
                    neg_content = "\n".join(final_negative_prompts)
                    
                    # Update message history if needed
                    if keep_alive and formatted_response:
                        messages.append({"role": "user", "content": user_prompt})
                        messages.append({"role": "assistant", "content": formatted_response})

                    return {
                        "Question": user_prompt,
                        "Response": formatted_response,
                        "Negative": neg_content,
                        "Tool_Output": None,
                        "Retrieved_Image": current_images,  # Return original images
                        "Mask": mask_tensor
                    }

                except Exception as e:
                    logger.error(f"Error in normal strategy: {str(e)}")
                    # Return original images or placeholder on error
                    if images is not None:
                        current_images = images  # Use original images
                        if mask is not None:
                            current_mask = mask
                        else:
                            # Create default mask matching image dimensions
                            current_mask = torch.ones((current_images.shape[0], 1, current_images.shape[2], current_images.shape[3]), 
                                                    dtype=torch.float32,
                                                    device=current_images.device)
                    else:
                        current_images, current_mask = load_placeholder_image(self.placeholder_image_path)

                    return {
                        "Question": user_prompt,
                        "Response": f"Error in processing: {str(e)}",
                        "Negative": "",
                        "Tool_Output": None,
                        "Retrieved_Image": current_images,
                        "Mask": current_mask 
                    }
                    
        except Exception as e:
            logger.error(f"Error in process_image: {str(e)}")
            return {
                "Question": kwargs.get("user_prompt", ""),
                "Response": f"Error: {str(e)}",
                "Negative": "",
                "Tool_Output": None,
                "Retrieved_Image": (
                    images[0]
                    if images is not None and len(images) > 0
                    else load_placeholder_image(self.placeholder_image_path)[0]
                ),
                "Mask": (
                    torch.ones_like(images[0][:1])
                    if images is not None and len(images) > 0
                    else load_placeholder_image(self.placeholder_image_path)[1]
                ),
            }

    def process_image_wrapper(self, **kwargs):
        """Wrapper to handle async execution of process_image"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            # Ensure images is present in kwargs
            if 'images' not in kwargs:
                raise ValueError("Input images are required")

            # Ensure all other required parameters are present
            required_params = ['llm_provider', 'llm_model', 'base_ip', 'port', 'user_prompt']
            missing_params = [p for p in required_params if p not in kwargs]
            if missing_params:
                raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")

            # Get the result from process_image
            result = loop.run_until_complete(self.process_image(**kwargs))

            # Extract values in the correct order matching RETURN_TYPES
            prompt = result.get("Response", "")  # This is the formatted prompt
            response = result.get("Question", "")  # Original question/prompt
            negative = result.get("Negative", "")
            omni = result.get("Tool_Output")
            retrieved_image = result.get("Retrieved_Image")
            mask = result.get("Mask")

            # Ensure we have valid image and mask tensors
            if retrieved_image is None or not isinstance(retrieved_image, torch.Tensor):
                retrieved_image, mask = load_placeholder_image(self.placeholder_image_path)

            # Ensure mask has correct format
            if mask is None:
                mask = torch.ones((retrieved_image.shape[0], 1, retrieved_image.shape[2], retrieved_image.shape[3]), 
                                dtype=torch.float32,
                                device=retrieved_image.device)

            # Return tuple matching RETURN_TYPES order: ("STRING", "STRING", "STRING", "OMNI", "IMAGE", "MASK")
            return (
                response,  # First STRING (question/prompt)
                prompt,    # Second STRING (generated response)
                negative,  # Third STRING (negative prompt)
                omni,      # OMNI
                retrieved_image,  # IMAGE
                mask       # MASK
            )

        except Exception as e:
            logger.error(f"Error in process_image_wrapper: {str(e)}")
            # Create fallback values
            image_tensor, mask_tensor = load_placeholder_image(self.placeholder_image_path)
            return (
                kwargs.get("user_prompt", ""),  # Original prompt
                f"Error: {str(e)}",            # Error message as response
                "",                            # Empty negative prompt
                None,                          # No OMNI data
                image_tensor,                  # Placeholder image
                mask_tensor                    # Default mask
            )

# Node registration
NODE_CLASS_MAPPINGS = {
    "IF_ImagePrompt": IFImagePrompt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IF_ImagePrompt": "IF Image to Prompt🖼️"
}
