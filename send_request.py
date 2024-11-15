#send_request.py
import aiohttp
import asyncio
import json
import logging
from typing import List, Union, Optional, Dict, Any
#from json_repair import repair_json

# Existing imports
from .anthropic_api import send_anthropic_request
from .ollama_api import send_ollama_request, create_ollama_embedding
from .openai_api import send_openai_request, create_openai_compatible_embedding, generate_image, generate_image_variations, edit_image
from .xai_api import send_xai_request
from .kobold_api import send_kobold_request
from .groq_api import send_groq_request
from .lms_api import send_lmstudio_request
from .textgen_api import send_textgen_request
from .llamacpp_api import send_llama_cpp_request
from .mistral_api import send_mistral_request 
from .vllm_api import send_vllm_request
from .gemini_api import send_gemini_request
from .transformers_api import TransformersModelManager  # Import the manager
from .utils import format_images_for_provider, convert_images_for_api, format_response
# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)    

# Initialize the TransformersModelManager
_transformers_manager = TransformersModelManager()  


def run_async(coroutine):
    """Helper function to run coroutines in a new event loop if necessary"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coroutine)

async def send_request(
    llm_provider: str,
    base_ip: str,
    port: str,
    images: List[str],
    llm_model: str,
    system_message: str,
    user_message: str,
    messages: List[Dict[str, Any]],
    seed: Optional[int],
    temperature: float,
    max_tokens: int,
    random: bool,
    top_k: int,
    top_p: float,
    repeat_penalty: float,
    stop: Optional[List[str]],
    keep_alive: bool,
    llm_api_key: Optional[str] = None,
    tools: Optional[Any] = None,
    tool_choice: Optional[Any] = None,
    precision: Optional[str] = "fp16", 
    attention: Optional[str] = "sdpa",
    aspect_ratio: Optional[str] = "1:1",
    strategy: Optional[str] = "normal",
    batch_count: Optional[int] = 4,
    mask: Optional[str] = None,
) -> Union[str, Dict[str, Any]]:
    """
    Sends a request to the specified LLM provider and returns a unified response.

    Args:
        llm_provider (str): The LLM provider to use.
        base_ip (str): Base IP address for the API.
        port (int): Port number for the API.
        base64_images (List[str]): List of images encoded in base64.
        llm_model (str): The model to use.
        system_message (str): System message for the LLM.
        user_message (str): User message for the LLM.
        messages (List[Dict[str, Any]]): Conversation messages.
        seed (Optional[int]): Random seed.
        temperature (float): Temperature for randomness.
        max_tokens (int): Maximum tokens to generate.
        random (bool): Whether to use randomness.
        top_k (int): Top K for sampling.
        top_p (float): Top P for sampling.
        repeat_penalty (float): Penalty for repetition.
        stop (Optional[List[str]]): Stop sequences.
        keep_alive (bool): Whether to keep the session alive.
        llm_api_key (Optional[str], optional): API key for the LLM provider.
        tools (Optional[Any], optional): Tools to be used.
        tool_choice (Optional[Any], optional): Tool choice.
        precision (Optional[str], optional): Precision for the model.
        attention (Optional[str], optional): Attention mechanism for the model.
        aspect_ratio (Optional[str], optional): Desired aspect ratio for image generation/editing. 
            Options: "1:1", "4:5", "3:4", "5:4", "16:9", "9:16". Defaults to "1:1".
        image_mode (Optional[str], optional): Mode for image processing.
            Options: "create", "edit", "variations". Defaults to "create".

    Returns:
        Union[str, Dict[str, Any]]: Unified response format.
    """
    try:
        #formatted_images = format_images_for_provider(images, llm_provider) if images is not None else None
        #formatted_mask = format_images_for_provider(mask, llm_provider) if mask is not None else None
        # Define aspect ratio to size mapping
        aspect_ratio_mapping = {
            "1:1": "1024x1024",
            "4:5": "1024x1280",
            "3:4": "1024x1365",
            "5:4": "1280x1024",
            "16:9": "1600x900",
            "9:16": "900x1600"
        }

        # Get the size based on the provided aspect_ratio
        size = aspect_ratio_mapping.get(aspect_ratio.lower(), "1024x1024")  # Default to square if invalid

        # Convert images to base64 format for API consumption
        if llm_provider == "transformers":
            
            formatted_images = convert_images_for_api(images, target_format='pil') if images is not None and len(images) > 0 else None
            response = await _transformers_manager.send_transformers_request(
                model_name=llm_model,
                system_message=system_message,
                user_message=user_message,
                messages=messages,
                max_new_tokens=max_tokens,
                images=formatted_images,  
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop_strings_list=stop,
                repetition_penalty=repeat_penalty,
                seed=seed,
                keep_alive=keep_alive,
                precision=precision,
                attention=attention
            )
            return response
        else:
            # For other providers, convert to base64 only if images exist
            formatted_images = convert_images_for_api(images, target_format='base64') if images is not None and len(images) > 0 else None
            formatted_mask = convert_images_for_api(mask, target_format='base64') if mask is not None and len(mask) > 0 else None
            
            api_functions = {
                "groq": send_groq_request,
                "anthropic": send_anthropic_request,
                "openai": send_openai_request,
                "xai": send_xai_request,
                "kobold": send_kobold_request,
                "ollama": send_ollama_request,
                "lmstudio": send_lmstudio_request,
                "textgen": send_textgen_request,
                "llamacpp": send_llama_cpp_request,
                "mistral": send_mistral_request,
                "vllm": send_vllm_request,
                "gemini": send_gemini_request,
                "transformers": None,  # Handled separately
            }

            if llm_provider not in api_functions and llm_provider != "transformers":
                raise ValueError(f"Invalid llm_provider: {llm_provider}")

            if llm_provider == "transformers":
                # This should be handled above, but included for safety
                raise ValueError("Transformers provider should be handled separately.")
            else:
                # Existing logic for other providers
                api_function = api_functions[llm_provider]
                # Prepare API-specific keyword arguments
                kwargs = {}
                
                if llm_provider == "ollama":
                    api_url = f"http://{base_ip}:{port}/api/chat"  
                    kwargs = dict(
                        api_url=api_url,
                        base64_images=formatted_images,  
                        model=llm_model,
                        system_message=system_message,
                        user_message=user_message,
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
                        tools=tools,
                        tool_choice=tool_choice,
                    )
                elif llm_provider in ["kobold", "lmstudio", "textgen", "llamacpp", "vllm"]:
                    api_url = f"http://{base_ip}:{port}/v1/chat/completions"
                    kwargs = {
                        "api_url": api_url,
                        "base64_images": formatted_images,  
                        "model": llm_model,
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
                    if llm_provider == "llamacpp":
                        kwargs.pop("tool_choice", None)
                    elif llm_provider == "vllm":
                        kwargs["api_key"] = llm_api_key
                elif llm_provider == "gemini":
                    kwargs = {
                        "base64_images": formatted_images,  
                        "model": llm_model,
                        "system_message": system_message,
                        "user_message": user_message,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "top_k": top_k,
                        "top_p": top_p,
                        "stop": stop,
                        "api_key": llm_api_key,
                        "tools": tools,
                        "tool_choice": tool_choice,
                    }      
                elif llm_provider == "openai":
                    if llm_model.startswith("dall-e"):
                        if strategy == "create":
                            # Generate image
                            generated_images = await generate_image(
                                prompt=user_message,
                                model=llm_model,
                                n=batch_count,
                                size=size,
                                api_key=llm_api_key
                            )
                            return {"images": generated_images}
                        elif strategy == "edit":

                            # Edit image
                            edited_images = await edit_image(
                                image_base64=formatted_images[0],
                                mask_base64=formatted_mask,
                                prompt=user_message,
                                model=llm_model,
                                n=batch_count,
                                size=size,
                                api_key=llm_api_key
                            )
                            return {"images": edited_images}
                        elif strategy == "variations":
                            # Generate variations
                            variations_images = await generate_image_variations(
                                image_base64=formatted_images[0],
                                model=llm_model,
                                n=batch_count,
                                size=size,
                                api_key=llm_api_key
                            )
                            return {"images": variations_images}
                    else:
                        api_url = f"https://api.openai.com/v1/chat/completions"
                        kwargs = {
                            "api_url": api_url,
                            "base64_images": formatted_images,  
                            "model": llm_model,
                            "system_message": system_message,
                            "user_message": user_message,
                            "messages": messages,
                            "api_key": llm_api_key,
                            "seed": seed if random else None,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "top_p": top_p,
                            "repeat_penalty": repeat_penalty,
                            "tools": tools,
                            "tool_choice": tool_choice,
                        }
                elif llm_provider == "xai":
                    api_url = f"https://api.x.ai/v1/chat/completions"
                    kwargs = {
                        "api_url": api_url,
                        "base64_images": formatted_images,  
                        "model": llm_model,
                        "system_message": system_message,
                        "user_message": user_message,
                        "messages": messages,
                        "api_key": llm_api_key,
                        "seed": seed if random else None,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "top_p": top_p,
                        "repeat_penalty": repeat_penalty,
                        "tools": tools,
                        "tool_choice": tool_choice,
                    }
                elif llm_provider == "anthropic":
                    kwargs = {
                        "api_key": llm_api_key,
                        "model": llm_model,
                        "system_message": system_message,
                        "user_message": user_message,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "base64_images": formatted_images,  
                        "tools": tools,
                        "tool_choice": tool_choice
                    }
                elif llm_provider == "groq":
                    kwargs = {
                        "base64_images": formatted_images,  
                        "model": llm_model,
                        "system_message": system_message,
                        "user_message": user_message,
                        "messages": messages,
                        "api_key": llm_api_key,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "top_p": top_p,
                        "tools": tools,
                        "tool_choice": tool_choice,
                    }
                elif llm_provider == "mistral":
                    kwargs = {
                        "base64_images": formatted_images, 
                        "model": llm_model,
                        "system_message": system_message,
                        "user_message": user_message,
                        "messages": messages,
                        "api_key": llm_api_key,
                        "seed": seed if random else None,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "top_p": top_p,
                        "tools": tools,
                        "tool_choice": tool_choice,     
                    }
                else:
                    raise ValueError(f"Unsupported llm_provider: {llm_provider}")
            
            response = await api_function(**kwargs)

            # Ensure response is properly awaited if it's a coroutine
            if asyncio.iscoroutine(response):
                response = await response

            if isinstance(response, dict):
                choices = response.get("choices", [])
                if choices and "content" in choices[0].get("message", {}):
                    content = choices[0]["message"]["content"]
                    if content.startswith("Error:"):
                        print(f"Error from {llm_provider} API: {content}")
            if tools:
                return response
            else:
                try:
                    return response["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError) as e:
                    error_msg = f"Error formatting response: {str(e)}"
                    logger.error(error_msg)
                    return {"choices": [{"message": {"content": error_msg}}]}

    except Exception as e:
        logger.error(f"Exception in send_request: {str(e)}", exc_info=True)
        return {"choices": [{"message": {"content": f"Exception: {str(e)}"}}]}

def format_response(response, tools):
    """Helper function to format the response consistently"""
    if tools:
        return response
    try:
        if isinstance(response, dict) and "choices" in response:
            return response["choices"][0]["message"]["content"]
        return response
    except (KeyError, IndexError, TypeError) as e:
        error_msg = f"Error formatting response: {str(e)}"
        logger.error(error_msg)
        return {"choices": [{"message": {"content": error_msg}}]}

async def create_embedding(embedding_provider: str, api_base: str, embedding_model: str, input: Union[str, List[str]], embedding_api_key: Optional[str] = None) -> Union[List[float], None]: # Correct return type hint
    if embedding_provider == "ollama":
        return await create_ollama_embedding(api_base, embedding_model, input)
    
    
    elif embedding_provider in ["openai", "lmstudio", "llamacpp", "textgen", "mistral", "xai"]:
        try:
            return await create_openai_compatible_embedding(api_base, embedding_model, input, embedding_api_key) # Try block for more precise error handling
        except ValueError as e:
            print(f"Error creating embedding: {e}")  
            return None # Return None on error
    
    else:
        raise ValueError(f"Unsupported embedding_provider: {embedding_provider}")
