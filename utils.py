import os
import io
import re
import yaml
import json
import torch
import torchvision
import cv2
import base64
import logging
import datetime
import requests
import numpy as np
from io import BytesIO
from aiohttp import web
from dotenv import load_dotenv
from PIL import Image, ImageOps, ImageSequence
from typing import Tuple, Optional, Dict, Union, List, Any
import node_helpers
import torch.nn.functional as F
from torchvision.transforms import functional as TF


from typing import Union, List, Tuple

logger = logging.getLogger(__name__)

def format_images_for_provider(images: Union[torch.Tensor, List[torch.Tensor]], provider: str) -> Union[List[str], List[Dict]]:
    """
    Format images according to each provider's requirements.
    
    Args:
        images: Tensor or list of tensors in [B,C,H,W] format
        provider: LLM provider name
    
    Returns:
        Formatted images ready for API consumption
    """
    try:
        # First ensure images are base64 encoded
        base64_images = convert_images_for_api(images, target_format='base64')
        
        if not base64_images:
            return None

        if provider == "ollama":
            # Ollama expects raw base64 strings in "images" field
            return base64_images
            
        elif provider == "anthropic":
            # Anthropic expects format with media_type
            return [{
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg", 
                    "data": img
                }
            } for img in base64_images]
            
        elif provider in ["openai", "kobold", "lmstudio", "textgen", "llamacpp", "groq"]:
            # These providers expect OpenAI-compatible format
            return [{
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img}"
                }
            } for img in base64_images]

        elif provider == "mistral":
            # Mistral expects similar format to OpenAI
            return [{
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img}"
                }
            } for img in base64_images]
        
        elif provider == "gemini":
            # Gemini expects specific MIME type format
            return [{
                "type": "image",
                "data": {
                    "mime_type": "image/jpeg",
                    "data": img
                }
            } for img in base64_images]
            
        elif provider == "transformers":
            # Transformers expects PIL images
            return convert_images_for_api(images, target_format='pil')
        
        else:
            # Default to base64 strings for unknown providers
            logger.warning(f"Unknown provider {provider}, returning raw base64")
            return base64_images

    except Exception as e:
        logger.error(f"Error formatting images for {provider}: {str(e)}")
        return None
    
def convert_images_for_api(images, target_format='tensor'):
    """
    Convert images to the specified format for API consumption.
    Supports conversion to: tensor, base64, pil
    """
    if images is None:
        return None
        
    # Handle tensor input with ComfyUI compatibility
    if isinstance(images, torch.Tensor):
        if images.dim() == 3:  # Single image
            images = images.unsqueeze(0)
        # Permute tensor to ComfyUI format (B, H, W, C) -> (B, C, H, W)
        images = images.permute(0, 3, 1, 2)
        
        if target_format == 'tensor':
            return images
        elif target_format == 'base64':
            return [tensor_to_base64(img) for img in images]
        elif target_format == 'pil':
            return [TF.to_pil_image(img) for img in images]
            
    # Handle base64 input
    if isinstance(images, str) or (isinstance(images, list) and all(isinstance(x, str) for x in images)):
        base64_list = [images] if isinstance(images, str) else images
        if target_format == 'base64':
            return base64_list
        
        # Convert base64 to PIL first
        pil_images = [base64_to_pil(b64) for b64 in base64_list]
        if target_format == 'pil':
            return pil_images
        elif target_format == 'tensor':
            tensors = [pil_to_tensor(img) for img in pil_images]
            return torch.stack(tensors).permute(0, 3, 1, 2)  # Maintain ComfyUI format
            
    # Handle PIL input
    if isinstance(images, (list, tuple)) and all(isinstance(x, Image.Image) for x in images):
        if target_format == 'pil':
            return images
        elif target_format == 'base64':
            return [pil_image_to_base64(img) for img in images]
        elif target_format == 'tensor':
            tensors = [pil_to_tensor(img) for img in images]
            return torch.stack(tensors).permute(0, 3, 1, 2)  # Maintain ComfyUI format
    
    raise ValueError(f"Unsupported image format or target format: {target_format}")

def convert_single_image(image, target_format):
    """Helper function to convert a single image"""
    if isinstance(image, str) and image.startswith('data:image'):
        # Convert base64 to PIL
        base64_data = image.split('base64,')[1]
        image_data = base64.b64decode(base64_data)
        image = Image.open(BytesIO(image_data))
    
    if target_format == 'pil':
        return image
    elif target_format == 'tensor':
        return pil_to_tensor(image)
    elif target_format == 'base64':
        return pil_image_to_base64(image)

def load_placeholder_image(placeholder_image_path):
        
        # Ensure the placeholder image exists
        if not os.path.exists(placeholder_image_path):
            # Create a proper RGB placeholder image
            placeholder = Image.new('RGB', (512, 512), color=(73, 109, 137))
            os.makedirs(os.path.dirname(placeholder_image_path), exist_ok=True)
            placeholder.save(placeholder_image_path)
        
        img = node_helpers.pillow(Image.open, placeholder_image_path)
        
        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']
        
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]
            
            if image.size[0] != w or image.size[1] != h:
                continue
            
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

def process_images_for_comfy(images, placeholder_image_path=None):
    """Process images for ComfyUI, ensuring consistent sizes."""
    def _process_single_image(image):
        try:
            if image is None:
                return load_placeholder_image(placeholder_image_path)

            # Convert to PIL Image first
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    image = image.squeeze(0)
                image = TF.to_pil_image(image)
            elif isinstance(image, np.ndarray):
                image = Image.fromarray((image * 255).astype(np.uint8))
            elif isinstance(image, str):
                if image.startswith(('data:image', 'http:', 'https:')):
                    if 'base64,' in image:
                        base64_data = image.split('base64,')[1]
                        image_data = base64.b64decode(base64_data)
                        image = Image.open(BytesIO(image_data))
                    else:
                        response = requests.get(image)
                        image = Image.open(BytesIO(response.content))
                else:
                    image = Image.open(image)

            # Ensure RGB mode
            if not isinstance(image, Image.Image):
                raise ValueError(f"Failed to convert to PIL Image: {type(image)}")
            
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Convert to tensor
            img_array = np.array(image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
            
            # Create mask
            mask_tensor = torch.ones((1, img_tensor.shape[2], img_tensor.shape[3]), dtype=torch.float32)
            
            return img_tensor, mask_tensor

        except Exception as e:
            logger.error(f"Error processing single image: {str(e)}")
            return load_placeholder_image(placeholder_image_path)

    try:
        if not isinstance(images, (list, tuple)):
            return _process_single_image(images)

        # Process all images
        processed = [_process_single_image(img) for img in images]
        if not processed:
            return _process_single_image(None)

        # Get target size from first image
        target_h, target_w = processed[0][0].shape[2:]
        
        # Resize all images to match first image
        resized = []
        for img_tensor, mask_tensor in processed:
            if img_tensor.shape[2:] != (target_h, target_w):
                img_tensor = F.interpolate(img_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)
                mask_tensor = F.interpolate(mask_tensor.unsqueeze(1), size=(target_h, target_w), mode='nearest').squeeze(1)
            resized.append((img_tensor, mask_tensor))

        # Combine all images and masks
        image_tensor = torch.cat([img for img, _ in resized], dim=0)
        mask_tensor = torch.cat([mask for _, mask in resized], dim=0)
        
        return image_tensor, mask_tensor

    except Exception as e:
        logger.error(f"Error in process_images_for_comfy: {str(e)}")
        return _process_single_image(None)

def process_mask(retrieved_mask, image_tensor):
    """
    Process the retrieved_mask to ensure it's in the correct format.
    The mask should be a tensor of shape (B, H, W), matching image_tensor's batch size and dimensions.
    """
    try:
        # Handle torch.Tensor
        if isinstance(retrieved_mask, torch.Tensor):
            # Normalize dimensions
            if retrieved_mask.dim() == 2:  # (H, W)
                retrieved_mask = retrieved_mask.unsqueeze(0)  # Add batch dimension
            elif retrieved_mask.dim() == 3:
                if retrieved_mask.shape[0] != image_tensor.shape[0]:
                    # Adjust batch size
                    retrieved_mask = retrieved_mask.repeat(image_tensor.shape[0], 1, 1)
            elif retrieved_mask.dim() == 4:
                # If mask has a channel dimension, reduce it
                retrieved_mask = retrieved_mask.squeeze(1)
            else:
                raise ValueError(f"Invalid mask tensor dimensions: {retrieved_mask.shape}")

            # Ensure proper format
            retrieved_mask = retrieved_mask.float()
            if retrieved_mask.max() > 1.0:
                retrieved_mask = retrieved_mask / 255.0

            # Ensure mask dimensions match image dimensions
            if retrieved_mask.shape[1:] != image_tensor.shape[2:]:
                # Resize mask to match image dimensions
                retrieved_mask = torch.nn.functional.interpolate(
                    retrieved_mask.unsqueeze(1),
                    size=(image_tensor.shape[2], image_tensor.shape[3]),
                    mode='nearest'
                ).squeeze(1)

            return retrieved_mask

        # Handle PIL Image
        elif isinstance(retrieved_mask, Image.Image):
            mask_array = np.array(retrieved_mask.convert('L')).astype(np.float32) / 255.0
            mask_tensor = torch.from_numpy(mask_array)
            mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dimension

            # Adjust batch size
            if mask_tensor.shape[0] != image_tensor.shape[0]:
                mask_tensor = mask_tensor.repeat(image_tensor.shape[0], 1, 1)

            # Resize if needed
            if mask_tensor.shape[1:] != image_tensor.shape[2:]:
                mask_tensor = torch.nn.functional.interpolate(
                    mask_tensor.unsqueeze(1),
                    size=(image_tensor.shape[2], image_tensor.shape[3]),
                    mode='nearest'
                ).squeeze(1)

            return mask_tensor

        # Handle numpy array
        elif isinstance(retrieved_mask, np.ndarray):
            mask_array = retrieved_mask.astype(np.float32)
            if mask_array.max() > 1.0:
                mask_array = mask_array / 255.0
            if mask_array.ndim == 2:
                pass  # (H, W)
            elif mask_array.ndim == 3:
                mask_array = np.mean(mask_array, axis=2)  # Convert to grayscale
            else:
                raise ValueError(f"Invalid mask array dimensions: {mask_array.shape}")

            mask_tensor = torch.from_numpy(mask_array)
            mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dimension

            # Adjust batch size
            if mask_tensor.shape[0] != image_tensor.shape[0]:
                mask_tensor = mask_tensor.repeat(image_tensor.shape[0], 1, 1)

            # Resize if needed
            if mask_tensor.shape[1:] != image_tensor.shape[2:]:
                mask_tensor = torch.nn.functional.interpolate(
                    mask_tensor.unsqueeze(1),
                    size=(image_tensor.shape[2], image_tensor.shape[3]),
                    mode='nearest'
                ).squeeze(1)

            return mask_tensor

        # Handle other types (e.g., file paths, base64 strings)
        elif isinstance(retrieved_mask, str):
            # Attempt to process as file path or base64 string
            if os.path.exists(retrieved_mask):
                pil_image = Image.open(retrieved_mask).convert('L')
            elif retrieved_mask.startswith('data:image'):
                base64_data = retrieved_mask.split('base64,')[1]
                image_data = base64.b64decode(base64_data)
                pil_image = Image.open(BytesIO(image_data)).convert('L')
            else:
                raise ValueError(f"Invalid mask string: {retrieved_mask}")
            return process_mask(pil_image, image_tensor)

        else:
            raise ValueError(f"Unsupported mask type: {type(retrieved_mask)}")

    except Exception as e:
        logger.error(f"Error processing mask: {str(e)}")
        # Return a default mask matching the image dimensions
        return torch.ones((image_tensor.shape[0], image_tensor.shape[2], image_tensor.shape[3]), dtype=torch.float32)

def convert_mask_to_grayscale_alpha(mask_input):
    """
    Convert mask to grayscale alpha channel.
    Handles tensors, PIL images and numpy arrays.
    Returns tensor in shape [B,1,H,W].
    """
    if isinstance(mask_input, torch.Tensor):
        # Handle tensor input
        if mask_input.dim() == 2:  # [H,W]
            return mask_input.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        elif mask_input.dim() == 3:  # [C,H,W] or [B,H,W]
            if mask_input.shape[0] in [1,3,4]:  # Assume channel-first
                if mask_input.shape[0] == 4:  # Use alpha channel
                    return mask_input[3:4].unsqueeze(0)
                else:  # Convert to grayscale
                    weights = torch.tensor([0.299, 0.587, 0.114]).to(mask_input.device)
                    return (mask_input * weights.view(-1,1,1)).sum(0).unsqueeze(0).unsqueeze(0)
            else:  # Assume batch dimension
                return mask_input.unsqueeze(1)  # Add channel dim
        elif mask_input.dim() == 4:  # [B,C,H,W]
            if mask_input.shape[1] == 4:  # Use alpha channel
                return mask_input[:,3:4]
            else:  # Convert to grayscale
                weights = torch.tensor([0.299, 0.587, 0.114]).to(mask_input.device)
                return (mask_input * weights.view(1,-1,1,1)).sum(1).unsqueeze(1)
                
    elif isinstance(mask_input, Image.Image):
        # Convert PIL image to grayscale
        mask = mask_input.convert('L')
        tensor = torch.from_numpy(np.array(mask)).float() / 255.0
        return tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
    elif isinstance(mask_input, np.ndarray):
        # Handle numpy array
        if mask_input.ndim == 2:  # [H,W]
            tensor = torch.from_numpy(mask_input).float()
            return tensor.unsqueeze(0).unsqueeze(0)
        elif mask_input.ndim == 3:  # [H,W,C]
            if mask_input.shape[2] == 4:  # Use alpha channel
                tensor = torch.from_numpy(mask_input[:,:,3]).float()
            else:  # Convert to grayscale
                tensor = torch.from_numpy(np.dot(mask_input[...,:3], [0.299, 0.587, 0.114])).float()
            return tensor.unsqueeze(0).unsqueeze(0)
            
    raise ValueError(f"Unsupported mask input type: {type(mask_input)}")

def tensor_to_base64(tensor: torch.Tensor) -> str:
    """Convert a tensor to a base64-encoded PNG image string."""
    try:
        # Ensure the tensor is in [0, 1] range
        tensor = torch.clamp(tensor, 0, 1)
        
        # If tensor has a channel dimension, ensure it's last
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        if tensor.dim() == 3:
            image = tensor.permute(1, 2, 0).cpu().numpy()  # [C, H, W] -> [H, W, C]
        elif tensor.dim() == 2:
            # For masks with shape [H, W], add a dummy channel
            image = tensor.unsqueeze(-1).cpu().numpy()      # [H, W] -> [H, W, 1]
        else:
            raise ValueError(f"Unsupported tensor shape for conversion: {tensor.shape}")

        # Handle single-channel images by converting them to RGB
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)  # [H, W, 1] -> [H, W, 3]

        # Convert to uint8
        image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
    except Exception as e:
        logger.error(f"Error converting tensor to base64: {str(e)}", exc_info=True)
        raise

def pil_to_tensor(pil_image):
    """Convert PIL image to tensor in ComfyUI format"""
    return torch.from_numpy(np.array(pil_image)).float() / 255.0

def base64_to_pil(base64_str):
    """Convert base64 string to PIL Image"""
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split('base64,')[1]
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data))

def pil_image_to_base64(pil_image: Image.Image) -> str:
    """Converts a PIL Image to a data URL."""
    try:
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        logger.error(f"Error converting image to data URL: {str(e)}", exc_info=True)
        raise

def clean_text(generated_text, remove_weights=True, remove_author=True):
    """Clean text while preserving intentional line breaks."""
    # Split into lines first to preserve breaks
    lines = generated_text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        if line.strip():  # Only process non-empty lines
            # Remove author attribution if requested
            if remove_author:
                line = re.sub(r"\bby:.*", "", line)

            # Remove weights if requested
            if remove_weights:
                line = re.sub(r"\(([^)]*):[\d\.]*\)", r"\1", line)
                line = re.sub(r"(\w+):[\d\.]*(?=[ ,]|$)", r"\1", line)

            # Remove markup tags
            line = re.sub(r"<[^>]*>", "", line)

            # Remove lonely symbols and formatting
            line = re.sub(r"(?<=\s):(?=\s)", "", line)
            line = re.sub(r"(?<=\s);(?=\s)", "", line)
            line = re.sub(r"(?<=\s),(?=\s)", "", line)
            line = re.sub(r"(?<=\s)#(?=\s)", "", line)

            # Clean up extra spaces while preserving line structure
            line = re.sub(r"\s{2,}", " ", line)
            line = re.sub(r"\.,", ",", line)
            line = re.sub(r",,", ",", line)

            # Remove audio tags from the line
            if "<audio" in line:
                print(f"iF_prompt_MKR: Audio has been generated.")
                line = re.sub(r"<audio.*?>.*?</audio>", "", line)

            cleaned_lines.append(line.strip())

    # Join with newlines to preserve line structure
    return "\n".join(cleaned_lines)

def get_api_key(api_key_name, engine):
    local_engines = ["ollama", "llamacpp", "kobold", "lmstudio", "textgen", "sentence_transformers", "transformers"]
    
    if engine.lower() in local_engines:
        print(f"You are using {engine} as the engine, no API key is required.")
        return "1234"
    
    # Try to get the key from .env first
    load_dotenv()
    api_key = os.getenv(api_key_name)
    
    if api_key:
        print(f"API key for {api_key_name} found in .env file")
        return api_key
    
    # If .env is empty, get the key from os.environ
    api_key = os.getenv(api_key_name)
    
    if api_key:
        print(f"API key for {api_key_name} found in environment variables")
        return api_key
    
    print(f"API key for {api_key_name} not found in .env file or environment variables")
    raise ValueError(f"{api_key_name} not found. Please set it in your .env file or as an environment variable.")

def get_models(engine, base_ip, port, api_key):

    if engine == "ollama":
        api_url = f"http://{base_ip}:{port}/api/tags"
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            models = [model["name"] for model in response.json().get("models", [])]
            return models
        except Exception as e:
            print(f"Failed to fetch models from Ollama: {e}")
            return []

    elif engine == "lmstudio":
        api_url = f"http://{base_ip}:{port}/v1/models"
        try:
            print(f"Attempting to connect to {api_url}")
            response = requests.get(api_url, timeout=10)
            print(f"Response status code: {response.status_code}")
            print(f"Response content: {response.text}")
            if response.status_code == 200:
                data = response.json()
                models = [model["id"] for model in data["data"]]
                return models
            else:
                print(f"Failed to fetch models from LM Studio. Status code: {response.status_code}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to LM Studio server: {e}")
            return []

    elif engine == "textgen":
        api_url = f"http://{base_ip}:{port}/v1/internal/model/list"
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            models = response.json()["model_names"]
            return models
        except Exception as e:
            print(f"Failed to fetch models from text-generation-webui: {e}")
            return []

    elif engine == "kobold":
        api_url = f"http://{base_ip}:{port}/api/v1/model"
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            model = response.json()["result"]
            return [model]
        except Exception as e:
            print(f"Failed to fetch models from Kobold: {e}")
            return []

    elif engine == "llamacpp":
        api_url = f"http://{base_ip}:{port}/v1/models"
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
            # Adapt this based on vLLM"s actual API response structure
            models = [model["id"] for model in response.json()["data"]] 
            return models
        except Exception as e:
            print(f"Failed to fetch models from vLLM: {e}")
            return []

    elif engine == "openai":
        fallback_models = [
            "tts-l-hd", "dall-e-3", "whisper-I", "text-embedding-3-large", 
            "text-embedding-3-small", "text-embedding-ada-002", "gpt-4-turbo", 
            "gpt-4-turbo-2024-04-09", "gpt-4-0125-preview", "gpt-3.5-turbo", 
            "gpt-4-turbo-preview", "gpt-4", "davinci-002", "gpt-4o-mini", 
            "gpt-4o", "gpt40-0806-loco-vm"
        ]

        #api_key = get_api_key("OPENAI_API_KEY", engine)
        if not api_key or api_key == "1234":
            print("Warning: Invalid OpenAI API key. Using fallback model list.")
            return fallback_models

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        api_url = "https://api.openai.com/v1/models"
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            api_models = [model["id"] for model in response.json()["data"]]
            print(f"Successfully fetched {len(api_models)} models from OpenAI API")
            
            # Combine API models with fallback models, prioritizing API models
            combined_models = list(set(api_models + fallback_models))
            return combined_models
        except Exception as e:
            print(f"Failed to fetch models from OpenAI: {e}")
            if isinstance(e, requests.exceptions.RequestException) and hasattr(e, "response"):
                print(f"Response status code: {e.response.status_code}")
                print(f"Response content: {e.response.text}")
            print(f"Returning fallback list of {len(fallback_models)} OpenAI models")
            return fallback_models
    
    elif engine == "xai":
        fallback_models = [
            "grok-beta"
        ]

        #api_key = get_api_key("XAI_API_KEY", engine)
        if not api_key or api_key == "1234":
            print("Warning: Invalid OpenAI API key. Using fallback model list.")
            return fallback_models

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        api_url = "https://api.x.ai/v1/models"
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            api_models = [model["id"] for model in response.json()["data"]]
            print(f"Successfully fetched {len(api_models)} models from XAI API")
            
            # Combine API models with fallback models, prioritizing API models
            combined_models = list(set(api_models + fallback_models))
            return combined_models
        except Exception as e:
            print(f"Failed to fetch models from XAI: {e}")
            if isinstance(e, requests.exceptions.RequestException) and hasattr(e, "response"):
                print(f"Response status code: {e.response.status_code}")
                print(f"Response content: {e.response.text}")
            print(f"Returning fallback list of {len(fallback_models)} XAI models")
            return fallback_models

    elif engine == "mistral":
        fallback_models = [
            "open-mistral-7b", "mistral-tiny", "mistral-tiny-2312",
            "open-mistral-nemo", "open-mistral-nemo-2407", "mistral-tiny-2407",
            "mistral-tiny-latest", "open-mixtral-8x7b", "mistral-small",
            "mistral-small-2312", "open-mixtral-8x22b", "open-mixtral-8x22b-2404",
            "mistral-small-2402", "mistral-small-latest", "mistral-medium-2312",
            "mistral-medium", "mistral-medium-latest", "mistral-large-2402",
            "mistral-large-2407", "mistral-large-latest", "codestral-2405",
            "codestral-latest", "codestral-mamba-2407", "open-codestral-mamba",
            "codestral-mamba-latest", "mistral-embed"
        ]

        #api_key = get_api_key("MISTRAL_API_KEY", engine)
        if not api_key or api_key == "1234":
            print("Warning: Invalid Mistral API key. Using fallback model list.")
            return fallback_models

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        api_url = "https://api.mistral.ai/v1/models"
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            api_models = [model["id"] for model in response.json()["data"]]
            print(f"Successfully fetched {len(api_models)} models from Mistral API")
            
            # Combine API models with fallback models, prioritizing API models
            combined_models = list(set(api_models + fallback_models))
            return combined_models
        except Exception as e:
            print(f"Failed to fetch models from Mistral: {e}")
            print(f"Returning fallback list of {len(fallback_models)} Mistral models")
            return fallback_models

    elif engine == "groq":
        fallback_models = [
            "llama-3.1-8b-instant",
            "llava-v1.5-7b-4096-preview",
            "gemma2-9b-it",
            "whisper-large-v3",
            "llama-3.1-70b-versatile",
            "llama3-groq-70b-8192-tool-use-preview",
            "llama3-groq-8b-8192-tool-use-preview",
            "llama-guard-3-8b",
            "llama3-70b-8192",
            "distil-whisper-large-v3-en",
            "mixtral-8x7b-32768",
            "llama3-8b-8192",
        ]

        #api_key = get_api_key("GROQ_API_KEY", engine)
        if not api_key or api_key == "1234":
            print("Warning: Invalid GROQ API key. Using fallback model list.")
            return fallback_models

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        api_url = "https://api.groq.com/openai/v1/models"
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            api_models = [model["id"] for model in response.json()["data"]]
            print(f"Successfully fetched {len(api_models)} models from GROQ API")
            
            # Combine API models with fallback models, prioritizing API models
            combined_models = list(set(api_models + fallback_models))
            return combined_models
        except Exception as e:
            print(f"Failed to fetch models from GROQ: {e}")
            print(f"Returning fallback list of {len(fallback_models)} GROQ models")
            return fallback_models

    elif engine == "anthropic":
        return [
            "claude-3-5-opus-latest",
            "claude-3-opus-20240229",
            "claude-3-5-sonnet-latest",
            "claude-3-5-sonnet-20240620",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-haiku-latest",
            "claude-3-5-haiku-20241022",
        ]

    elif engine == "gemini":
        return [
            "gemini-exp-1114",
            "gemini-1.5-pro-002",
            "gemini-1.5-flash-002",
            "gemini-1.5-flash-8b-exp-0924",
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash",
            "gemini-1.5-pro-latest",
            "gemini-1.5-latest",
            "gemini-pro",
            "gemini-pro-vision",
        ]

    elif engine == "sentence_transformers":
        return [
            "sentence-transformers/all-MiniLM-L6-v2",
            "avsolatorio/GIST-small-Embedding-v0",
        ]

    elif engine == "transformers":
        return [            
            "impactframes/Llama-3.2-11B-Vision-bnb-4bit",
            "impactframes/pixtral-12b-4bit",
            "impactframes/molmo-7B-D-bnb-4bit",
            "impactframes/molmo-7B-O-bnb-4bit",
            "impactframes/Qwen2-VL-7B-Captioner",
            "impactframes/colqwen2-v0.1",
            "impactframes/colpali-v1.2",
            "impactframes/Florence-2-DocVQA",
            "vidore/colpali",
            "Qwen/Qwen2-7B-Instruct",
            "Qwen/Qwen2-VL-2B-Instruct",
            "microsoft/Florence-2-base",
            "microsoft/Florence-2-base-ft",
            "microsoft/Florence-2-large",
            "microsoft/Florence-2-large-ft",
        ]

    else:
        print(f"Unsupported engine - {engine}")
        return []

def validate_models(model, provider, model_type, base_ip, port, api_key):
        available_models = get_models(provider, base_ip, port, api_key)
        if available_models is None or model not in available_models:
            error_message = f"Invalid {model_type} model selected: {model} for provider {provider}. Available models: {available_models}"
            print(error_message)
            raise ValueError(error_message)

class EnhancedYAMLDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(EnhancedYAMLDumper, self).increase_indent(flow, False)

def str_presenter(dumper, data):
    if len(data.splitlines()) > 1:  # check for multiline string
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)

EnhancedYAMLDumper.add_representer(str, str_presenter)

def numpy_int64_presenter(dumper, data):
    return dumper.represent_int(int(data))

EnhancedYAMLDumper.add_representer(np.int64, numpy_int64_presenter)

def dump_yaml(data, file_path):
    """
    Safely dumps a dictionary to a YAML file with custom formatting.
    Converts any numpy.int64 values to int to avoid YAML serialization errors.
    Uses multi-line string representation for better readability.
    """
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Convert numpy types in the entire data structure
    data = yaml.safe_load(yaml.dump(data, default_flow_style=False, allow_unicode=True))

    with open(file_path, "w") as yaml_file:
        yaml.dump(data, yaml_file, Dumper=EnhancedYAMLDumper, default_flow_style=False, 
                  sort_keys=False, allow_unicode=True, width=1000, indent=2)

# Example usage
# sample_data = {
#     "key1": "Single line value",
#     "key2": "Multi-line\nvalue\nhere",
#     "key3": np.int64(42),
#     "key4": np.array([1, 2, 3])
# }
# dump_yaml(sample_data, "output.yaml")

def format_response(self, response):
        """
        Format the response by adding appropriate line breaks and paragraph separations.
        """
        paragraphs = re.split(r"\n{2,}", response)

        formatted_paragraphs = []
        for para in paragraphs:
            if "```" in para:
                parts = para.split("```")
                for i, part in enumerate(parts):
                    if i % 2 == 1:  # This is a code block
                        parts[i] = f"\n```\n{part.strip()}\n```\n"
                para = "".join(parts)
            else:
                para = para.replace(". ", ".\n")

            formatted_paragraphs.append(para.strip())

        return "\n\n".join(formatted_paragraphs)
