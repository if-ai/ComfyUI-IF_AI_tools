# transformers_api.py
from transformers import (
    Qwen2VLForConditionalGeneration, 
    Qwen2VLProcessor,
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    GenerationConfig,
    StopStringCriteria,
    set_seed,
)
from typing import List, Union, Optional, Dict, Any
from PIL import Image
from io import BytesIO
import base64
import torch
import logging
import os
import re
import folder_paths
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
import json
import importlib
import importlib.util 
import comfy.model_management as mm
from torchvision.transforms import functional as TF

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TransformersModelManager:
    def __init__(self):
        self.models_dir = os.path.join(folder_paths.models_dir, "LLM")
        self.models = {}
        self.processors = {}
        self.loaded_models = {} 
        self.device = mm.get_torch_device()
        self.offload_device = mm.unet_offload_device()
        self.model_path = None
        self.model_load_args = {
            "device_map": self.device,
            "torch_dtype": "auto", 
            "trust_remote_code": True
        }

    def download_model_if_not_exists(self, model_name):
        from huggingface_hub import snapshot_download

        model_dir = model_name.rsplit('/', 1)[-1]
        model_path = os.path.join(self.models_dir, model_dir)
        if not os.path.exists(model_path):
            logger.info(f"Downloading model '{model_name}' to: {model_path}")
            try:
                snapshot_download(
                    repo_id=model_name,
                    local_dir=model_path,
                    local_dir_use_symlinks=False,
                    token=os.getenv("HUGGINGFACE_TOKEN") or ""
                )
                logger.info(f"Model '{model_name}' downloaded successfully.")
            except Exception as e:
                logger.error(f"An error occurred while downloading the model '{model_name}': {e}")
                return None
        else:
            logger.info(f"Model '{model_name}' already exists at: {model_path}")
        return model_path

    def hash_seed(self, seed):
        import hashlib
        seed_bytes = str(seed).encode('utf-8')
        hash_object = hashlib.sha256(seed_bytes)
        hashed_seed = int(hash_object.hexdigest(), 16)
        return hashed_seed % (2**32)

    def load_model(self, model: str, precision: str, attention: str) -> Optional[Dict[str, Any]]:
        if model in self.loaded_models:
            logger.info(f"Model '{model}' already loaded and cached.")
            return self.loaded_models[model]

        if precision == "int8": 
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            dtype = torch.bfloat16 if 'mpt' in model.lower() or 'llama2' in model.lower() else torch.float16
        elif precision == "int4":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            dtype = torch.bfloat16
        else:
            quant_config = None
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(precision, torch.float16)

        model_path = self.download_model_if_not_exists(model)
        if model_path is None:
            logger.error(f"Model path for '{model}' could not be determined.")
            return None

        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            logger.error(f"Config file not found at: {config_path}")
            return None

        device = self.device
        try:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            architectures = config.architectures

            if architectures and isinstance(architectures, list) and len(architectures) > 0:
                model_class = architectures[0]
                try:
                    common_args = {
                        "pretrained_model_name_or_path": model_path,
                        "attn_implementation": attention,
                        "torch_dtype": dtype,
                        "trust_remote_code": True,
                        "device_map": device,
                    }

                    if quant_config:
                        common_args["quantization_config"] = quant_config

                    if "florence" in model.lower() or 'florence' in model_path.lower() or "deepseek" in model.lower() or 'deepseek' in model_path.lower():
                        with patch("transformers.dynamic_module_utils.get_imports", self.fixed_get_imports):
                            loaded_model = AutoModelForCausalLM.from_pretrained(**common_args)
                    elif "pixtral" in model.lower():
                        from transformers import LlavaForConditionalGeneration
                        loaded_model = LlavaForConditionalGeneration.from_pretrained(**common_args, use_safetensors=True)
                    elif "molmo" in model.lower():
                        loaded_model = AutoModelForCausalLM.from_pretrained(**common_args, use_safetensors=True)
                    elif "qwen2-vl" in model.lower():
                        min_pixels = 224 * 224
                        max_pixels = 1024 * 1024
                        processor = Qwen2VLProcessor.from_pretrained(
                            model_path,
                            min_pixels=min_pixels, 
                            max_pixels=max_pixels,
                            trust_remote_code=True
                        )
                        loaded_model = Qwen2VLForConditionalGeneration.from_pretrained(**common_args, use_safetensors=True)
                    else:
                        loaded_model = model_class.from_pretrained(**common_args)

                    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

                except AttributeError:
                    logger.warning(f"AttributeError encountered. Forcing trust_remote_code=True for model: {model}")
                    loaded_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map=device)
                    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        except Exception as e:
            logger.error(f"Error loading model from config.json: {e}")
            return None

        self.loaded_models[model] = {'model': loaded_model, 'processor': processor, 'dtype': dtype}
        logger.info(f"Model '{model}' loaded successfully and cached.")
        return self.loaded_models[model]

    async def send_transformers_request(
        self,
        model_name,
        system_message,
        user_message,
        messages,
        max_new_tokens,
        images,
        temperature,
        top_p,
        top_k,
        stop_strings_list,
        repetition_penalty,
        seed,
        keep_alive=True,
        precision="fp16",
        attention="sdpa",
    ):
        try:
            if model_name in self.loaded_models:
                logger.info(f"Model '{model_name}' already loaded and cached.")
                model_data = self.loaded_models[model_name]
            else:
                model_data = self.load_model(model_name, precision=precision, attention=attention)  
                if model_data is None:
                    raise ValueError(f"Failed to load model '{model_name}'.")

            model = model_data['model']
            processor = model_data['processor']
            tokenizer = processor.tokenizer
            dtype = model_data['dtype']

            if seed is not None:
                logger.info(f"Setting seed: {seed}")
                set_seed(self.hash_seed(seed))

            # Convert to PIL Images if necessary
            pil_images = []
            if isinstance(images, torch.Tensor):
                images = images.permute(0, 3, 1, 2)
                for img in images:
                    pil_images.append(TF.to_pil_image(img))
            elif isinstance(images, list) and all(isinstance(img, Image.Image) for img in images):
                pil_images = images
            else:
                raise ValueError("Images must be either a torch.Tensor or a list of PIL Images")

            logger.debug(f"Number of images processed: {len(pil_images)}")

            # Construct standardized messages
            formatted_messages = self.construct_messages(system_message, user_message, messages, pil_images)

            logger.debug(f"Formatted messages: {formatted_messages}")

            if 'florence' in model_name.lower():
                # Process input for Florence models
                generated_texts = []
                responses = []
                images_pil = []
                for pil_image in pil_images:
                    inputs = processor(images=[pil_image], text=user_message, return_tensors="pt", do_rescale=False).to(dtype).to(model.device)

                    logger.debug(f"Inputs shape: {inputs['pixel_values'].shape}, dtype: {inputs['pixel_values'].dtype}")
                    logger.debug(f"Input IDs shape: {inputs['input_ids'].shape}, dtype: {inputs['input_ids'].dtype}")

                    with torch.random.fork_rng(devices=[model.device]):
                        torch.random.manual_seed(seed)

                        try:
                            generated_ids = model.generate(
                                **inputs,
                                max_new_tokens=max_new_tokens,
                                num_beams=3,
                                do_sample=True,
                                temperature=temperature,
                                top_p=top_p,
                                top_k=top_k,
                                repetition_penalty=repetition_penalty,
                            )
                        except Exception as e:
                            logger.error(f"Error during model.generate: {e}")
                            raise

                    results = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                    generated_text = self.clean_results(results, user_message)
                    response = processor.post_process_generation(generated_text, task=user_message, image_size=pil_image.size)
                    generated_texts.append(generated_text)
                    responses.append(response)
                    # images_pil.append(pil_image)

                result = (generated_texts, responses)
            else:
                # Handle other transformers models
                inputs = processor(formatted_messages, return_tensors="pt", padding=True).to(model.device)

                # Convert inputs to the correct dtype
                inputs = {k: v.to(dtype=torch.long if v.dtype == torch.int64 else dtype) if torch.is_tensor(v) else v for k, v in inputs.items()}

                with torch.no_grad():
                    try:
                        outputs = model.generate(
                            **inputs, 
                            generation_config=GenerationConfig(
                                max_new_tokens=max_new_tokens,
                                do_sample=True,
                                temperature=temperature,
                                top_p=top_p,
                                top_k=top_k,
                                repetition_penalty=repetition_penalty,
                            ),
                            stopping_criteria=[StopStringCriteria(tokenizer=tokenizer, stop_strings=stop_strings_list)],
                        )
                    except Exception as e:
                        logger.error(f"Error during model.generate: {e}")
                        raise

                generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

                result = generated_text

            if not keep_alive:
                self.unload_model(model_name)

            return result

        except Exception as e:
            logger.error(f"Error in Transformers API request: {e}", exc_info=True)
            return str(e)

    def clean_results(self, results, task):
        if task == 'ocr_with_region':
            clean_results = re.sub(r'</?s>|<[^>]*>', '\n', results)
            clean_results = re.sub(r'\n+', '\n', clean_results)
        else:
            clean_results = results.replace('</s>', '').replace('<s>', '')
        return clean_results

    def construct_messages(self, system_message, user_message, messages, pil_images):
        """Constructs a standardized message format for transformer models."""
        formatted_messages = []
        if system_message:
            formatted_messages.append({"role": "system", "content": system_message})

        for msg in messages:
            formatted_messages.append({"role": msg['role'], "content": msg['content']})

        if user_message:
            formatted_messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    *[{"type": "image", "image": img} for img in pil_images]
                ]
            })

        return formatted_messages

    def unload_model(self, model_name: str):
        print(f"Offloading model: {model_name}")
        if model_name in self.loaded_models:
            model = self.loaded_models[model_name]['model']
            model.to(self.offload_device)
            del self.loaded_models[model_name]
            mm.soft_empty_cache()
        else:
            print(f"Model {model_name} not found in loaded models.")

    @classmethod
    def fixed_get_imports(cls, filename: Union[str, os.PathLike], *args, **kwargs) -> List[str]:
        """Remove 'flash_attn' from imports if present."""
        try:
            if not str(filename).endswith("modeling_florence2.py") or not str(filename).endswith("modeling_deepseek.py"):
                return get_imports(filename)
            imports = get_imports(filename)
            if "flash_attn" in imports:
                imports.remove("flash_attn")
            return imports
        except Exception as e:
            print(f"No flash_attn import to remove: {e}")
            return get_imports(filename)


# Initialize a global manager instance
_transformers_manager = TransformersModelManager()
