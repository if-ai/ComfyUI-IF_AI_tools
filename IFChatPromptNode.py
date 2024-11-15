# IFChatPromptNode.py
import os
import sys
import json
import torch
import shutil
import base64
import platform
import importlib
import subprocess
import numpy as np
import folder_paths
from PIL import Image
import yaml
from io import BytesIO
import asyncio
from typing import List, Union, Dict, Any, Tuple, Optional
from .agent_tool import AgentTool
from .send_request import send_request
#from .transformers_api import TransformersModelManager 
import tempfile
import threading
from aiohttp import web
from .graphRAG_module import GraphRAGapp
from .colpaliRAG_module import colpaliRAGapp
from .superflorence import FlorenceModule
from .utils import get_api_key, get_models, validate_models, clean_text, process_mask, load_placeholder_image, process_images_for_comfy
#from byaldi import RAGMultiModalModel 
# Set up logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# Add the ComfyUI directory to the Python path
comfy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, comfy_path)

ifchat_prompt_node = None

try:
    from server import PromptServer

    @PromptServer.instance.routes.post("/IF_ChatPrompt/get_llm_models")
    async def get_llm_models_endpoint(request):
        data = await request.json()
        llm_provider = data.get("llm_provider")
        engine = llm_provider
        base_ip = data.get("base_ip")
        port = data.get("port")
        external_api_key = data.get("external_api_key")
        
        logger.debug(f"Received request for LLM models. Provider: {llm_provider}, External API key provided: {bool(external_api_key)}")

        if external_api_key:
            api_key = external_api_key
            logger.debug("Using provided external LLM API key")
        else:
            api_key_name = f"{llm_provider.upper()}_API_KEY"
            try:
                api_key = get_api_key(api_key_name, engine)
                logger.debug("Using API key from environment or .env file")
            except ValueError:
                logger.warning(f"No API key found for {llm_provider}. Attempting to proceed without an API key.")
                api_key = None

        models = get_models(engine, base_ip, port, api_key)
        logger.debug(f"Fetched {len(models)} models for {llm_provider}")
        return web.json_response(models)

    @PromptServer.instance.routes.post("/IF_ChatPrompt/get_embedding_models")
    async def get_embedding_models_endpoint(request):
        data = await request.json()
        embedding_provider = data.get("embedding_provider")
        engine = embedding_provider
        base_ip = data.get("base_ip")
        port = data.get("port")
        external_api_key = data.get("external_api_key")
        
        logger.debug(f"Received request for LLM models. Provider: {embedding_provider}, External API key provided: {bool(external_api_key)}")

        if external_api_key:
            api_key = external_api_key
            logger.debug("Using provided external LLM API key")
        else:
            api_key_name = f"{embedding_provider.upper()}_API_KEY"
            try:
                api_key = get_api_key(api_key_name, engine)
                logger.debug("Using API key from environment or .env file")
            except ValueError:
                logger.warning(f"No API key found for {embedding_provider}. Attempting to proceed without an API key.")
                api_key = None

        models = get_models(engine, base_ip, port, api_key)
        logger.debug(f"Fetched {len(models)} models for {embedding_provider}")
        return web.json_response(models)

    @PromptServer.instance.routes.post("/IF_ChatPrompt/upload_file")
    async def upload_file_route(request):
        try:
            reader = await request.multipart()

            rag_folder_name = None
            file_content = None
            filename = None

            # Process all parts of the multipart request
            while True:
                part = await reader.next()
                if part is None:
                    break
                if part.name == "rag_root_dir":
                    rag_folder_name = await part.text()
                elif part.filename:
                    filename = part.filename
                    file_content = await part.read()

            if not filename or not file_content or not rag_folder_name:
                return web.json_response({"status": "error", "message": "Missing file, filename, or RAG folder name"})

            node = IFChatPrompt()
            input_dir = os.path.join(node.rag_dir, rag_folder_name, "input")

            if not os.path.exists(input_dir):
                os.makedirs(input_dir, exist_ok=True)

            file_path = os.path.join(input_dir, filename)

            with open(file_path, 'wb') as f:
                f.write(file_content)

            logger.info(f"File uploaded to: {file_path}")
            return web.json_response({"status": "success", "message": f"File uploaded to: {file_path}"})

        except Exception as e:
            logger.error(f"Error in upload_file_route: {str(e)}")
            return web.json_response({"status": "error", "message": f"Error uploading file: {str(e)}"})

    @PromptServer.instance.routes.post("/IF_ChatPrompt/setup_and_initialize")
    async def setup_and_initialize(request):
        global ifchat_prompt_node
        
        data = await request.json()
        folder_name = data.get('folder_name', 'rag_data')
        
        if ifchat_prompt_node is None:
            ifchat_prompt_node = IFChatPrompt()
        
        init_result = await ifchat_prompt_node.graphrag_app.setup_and_initialize_folder(folder_name, data)
        
        ifchat_prompt_node.rag_folder_name = folder_name
        ifchat_prompt_node.colpali_app.set_rag_root_dir(folder_name)   
        
        return web.json_response(init_result)

    @PromptServer.instance.routes.post("/IF_ChatPrompt/run_indexer")
    async def run_indexer_endpoint(request):
        try:
            data = await request.json()
            logger.debug(f"Received indexing request with data: {data}")

            global ifchat_prompt_node  # Access the global instance

            # Set the rag_root_dir in both modules using the global instance
            ifchat_prompt_node.graphrag_app.set_rag_root_dir(data.get('rag_folder_name'))
            ifchat_prompt_node.colpali_app.set_rag_root_dir(data.get('rag_folder_name'))

            query_type = data.get('mode_type')
            logger.debug(f"Query type: {query_type}")

            logger.debug(f"Starting indexing process for query type: {query_type}")

            # Initialize the colpali_model before calling insert, using the global instance
            if query_type == 'colpali' or query_type == 'colqwen2' or query_type == 'colpali-v1.2':
                _ = ifchat_prompt_node.colpali_app.get_colpali_model(query_type)  # This will load or retrieve the cached model
                result = await ifchat_prompt_node.colpali_app.insert()
            else:
                result = await ifchat_prompt_node.graphrag_app.insert()

            logger.debug(f"Indexing process completed with result: {result}")

            if result:
                return web.json_response({"status": "success", "message": f"Indexing complete for {query_type}"})
            else:
                return web.json_response({"status": "error", "message": "Indexing failed. Check server logs."}, status=500)

        except Exception as e:
            logger.error(f"Error in run_indexer_endpoint: {str(e)}")
            return web.json_response({"status": "error", "message": f"Error during indexing: {str(e)}"}, status=500)
        
    @PromptServer.instance.routes.post("/IF_ChatPrompt/process_chat")
    async def process_chat_endpoint(request):
        try:
            data = await request.json()
            
            # Set default values for required arguments if not provided
            defaults = {
                "prompt": "",
                "assistant": "Cortana",  # Default assistant
                "neg_prompt": "Default",  # Default negative prompt
                "embellish_prompt": "Default",  # Default embellishment
                "style_prompt": "Default",  # Default style
                "llm_provider": "ollama",
                "llm_model": "",
                "base_ip": "localhost",
                "port": "11434",
                "embedding_model": "",
                "embedding_provider": "sentence_transformers"
            }
            
            # Update data with defaults for missing keys
            for key, default_value in defaults.items():
                if key not in data:
                    data[key] = default_value
                    
            global ifchat_prompt_node 
            result = await ifchat_prompt_node.process_chat(**data)
            
            return web.json_response(result)
            
        except Exception as e:
            logger.error(f"Error in process_chat_endpoint: {str(e)}")
            return web.json_response({
                "status": "error",
                "message": f"Error processing chat: {str(e)}",
                "Question": data.get("prompt", ""),
                "Response": f"Error: {str(e)}",
                "Negative": "",
                "Tool_Output": None,
                "Retrieved_Image": None,
                "Mask": None
            }, status=500)

    @PromptServer.instance.routes.post("/IF_ChatPrompt/load_index")
    async def load_index_route(request):
        try:
            data = await request.json()
            index_name = data.get('rag_folder_name')
            query_type = data.get('query_type')
            
            if not index_name:
                logger.error("No index name provided in the request.")
                return web.json_response({
                    "status": "error", 
                    "message": "No index name provided"
                })

            # Check if index exists in .byaldi directory
            byaldi_index_path = os.path.join(".byaldi", index_name)
            if not os.path.exists(byaldi_index_path):
                logger.error(f"Index not found in .byaldi: {byaldi_index_path}")
                return web.json_response({
                    "status": "error",
                    "message": f"Index {index_name} does not exist"
                })

            try:
                global ifchat_prompt_node
                if ifchat_prompt_node is None:
                    logger.debug("Initializing IFChatPrompt instance.")
                    ifchat_prompt_node = IFChatPrompt()

                if query_type in ['colpali', 'colqwen2', 'colpali-v1.2']:
                    logger.debug(f"Loading model for query type: {query_type}")
                    
                    # Clear any existing cached index
                    ifchat_prompt_node.colpali_app.cleanup_index()
                    
                    # First get the base model
                    colpali_model = ifchat_prompt_node.colpali_app.get_colpali_model(query_type)
                    
                    if colpali_model:
                        # Load and cache the new index
                        model = await ifchat_prompt_node.colpali_app._prepare_model(query_type, index_name)
                        if not model:
                            raise ValueError("Failed to load and cache index")
                        
                        # Set the RAG root directory
                        ifchat_prompt_node.colpali_app.set_rag_root_dir(index_name)
                        
                        logger.info(f"Successfully loaded and cached index: {index_name}")
                        return web.json_response({
                            "status": "success",
                            "message": f"Successfully loaded index: {index_name}",
                            "rag_root_dir": index_name
                        })
                    else:
                        logger.error("Failed to initialize ColPali model.")
                        raise ValueError("Failed to initialize ColPali model")
                
                else:
                    logger.error(f"Unsupported query type: {query_type}")
                    return web.json_response({
                        "status": "error",
                        "message": f"Query type {query_type} not supported for loading indexes"
                    })

            except Exception as e:
                logger.error(f"Error loading index {index_name}: {str(e)}")
                return web.json_response({
                    "status": "error",
                    "message": f"Error loading index: {str(e)}"
                })

        except Exception as e:
            logger.error(f"Error in load_index_route: {str(e)}")
            return web.json_response({
                "status": "error",
                "message": f"Error processing request: {str(e)}"
            })

    # Add this with the other routes
    @PromptServer.instance.routes.post("/IF_ChatPrompt/delete_index")
    async def delete_index_route(request):
        try:
            data = await request.json()
            index_name = data.get('rag_folder_name')
            
            if not index_name:
                return web.json_response({
                    "status": "error", 
                    "message": "No index name provided"
                })

            # Path to the index
            index_path = os.path.join(".byaldi", index_name)
            
            if not os.path.exists(index_path):
                return web.json_response({
                    "status": "error",
                    "message": f"Index {index_name} does not exist"
                })

            # Delete the index directory
            try:
                shutil.rmtree(index_path)
                logger.info(f"Successfully deleted index: {index_name}")
                return web.json_response({
                    "status": "success",
                    "message": f"Successfully deleted index: {index_name}"
                })
            except Exception as e:
                logger.error(f"Error deleting index {index_name}: {str(e)}")
                return web.json_response({
                    "status": "error",
                    "message": f"Error deleting index: {str(e)}"
                })

        except Exception as e:
            logger.error(f"Error in delete_index_route: {str(e)}")
            return web.json_response({
                "status": "error",
                "message": f"Error processing request: {str(e)}"
            })

except AttributeError:
    print("PromptServer.instance not available. Skipping route decoration for IF_ChatPrompt.")

class IFChatPrompt:

    def __init__(self):
        self.base_ip = "localhost"
        self.port = "11434"
        self.llm_provider = "ollama"
        self.embedding_provider = "sentence_transformers"
        self.llm_model = ""
        self.embedding_model = ""
        self.assistant = "None"
        self.random = False

        self.comfy_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.rag_dir = os.path.join(folder_paths.base_path, "custom_nodes", "ComfyUI-IF_AI_tools", "IF_AI", "rag")
        self.presets_dir = os.path.join(folder_paths.base_path, "custom_nodes", "ComfyUI-IF_AI_tools", "IF_AI", "presets")
        
        self.stop_file = os.path.join(self.presets_dir, "stop_strings.json")
        self.assistants_file = os.path.join(self.presets_dir, "assistants.json")
        self.neg_prompts_file = os.path.join(self.presets_dir, "neg_prompts.json")
        self.embellish_prompts_file = os.path.join(self.presets_dir, "embellishments.json")
        self.style_prompts_file = os.path.join(self.presets_dir, "style_prompts.json")
        self.tasks_file = os.path.join(self.presets_dir, "florence_prompts.json")
        self.agents_dir = os.path.join(self.presets_dir, "agents")

        self.agent_tools = self.load_agent_tools()
        self.stop_strings = self.load_presets(self.stop_file)
        self.assistants = self.load_presets(self.assistants_file)
        self.neg_prompts = self.load_presets(self.neg_prompts_file)
        self.embellish_prompts = self.load_presets(self.embellish_prompts_file)
        self.style_prompts = self.load_presets(self.style_prompts_file)
        self.florence_prompts = self.load_presets(self.tasks_file)

        self.keep_alive = False
        self.seed = 94687328150
        self.messages = []
        self.history_steps = 10
        self.external_api_key = ""
        self.tool_input = ""
        self.prime_directives = None
        self.rag_folder_name = "rag_data"
        self.graphrag_app = GraphRAGapp()
        self.colpali_app = colpaliRAGapp()
        self.fix_json = True
        self.cached_colpali_model = None
        self.florence_app = FlorenceModule()
        self.florence_models = {}
        self.query_type = "global"  
        self.enable_RAG = False
        self.clear_history = False
        self.mode = False
        self.tool = "None"
        self.preset = "Default"
        self.precision = "fp16"
        self.task = None  
        self.attention = "sdpa" 
        self.aspect_ratio = "16:9"
        self.top_k_search = 3
        
        self.placeholder_image_path = os.path.join(folder_paths.base_path, "custom_nodes", "ComfyUI-IF_AI_tools", "IF_AI", "placeholder.png")

        if not os.path.exists(self.placeholder_image_path):
            placeholder = Image.new('RGB', (512, 512), color=(73, 109, 137))
            os.makedirs(os.path.dirname(self.placeholder_image_path), exist_ok=True)
            placeholder.save(self.placeholder_image_path)

    def load_presets(self, file_path):
        with open(file_path, 'r') as f:
            presets = json.load(f)
        return presets

    def load_agent_tools(self):
        os.makedirs(self.agents_dir, exist_ok=True)
        agent_tools = {}
        try:
            for filename in os.listdir(self.agents_dir):
                if filename.endswith('.json'):
                    full_path = os.path.join(self.agents_dir, filename)
                    with open(full_path, 'r') as f:
                        try:
                            data = json.load(f)
                            if 'output_type' not in data:
                                data['output_type'] = None
                            agent_tool = AgentTool(**data)
                            agent_tool.load()
                            if agent_tool._class_instance is not None:
                                if agent_tool.python_function:
                                    agent_tools[agent_tool.name] = agent_tool
                                else:
                                    print(f"Warning: Agent tool {agent_tool.name} in {filename} does not have a python_function defined.")
                            else:
                                print(f"Failed to create class instance for {filename}")
                        except json.JSONDecodeError:
                            print(f"Error: Invalid JSON in {filename}")
                        except Exception as e:
                            print(f"Error loading {filename}: {str(e)}")
            return agent_tools
        except Exception as e:
            print(f"Warning: Error accessing agent tools directory: {str(e)}")
            return {}

    async def process_chat(
        self,
        prompt,
        llm_provider,
        llm_model,
        base_ip,
        port,
        assistant,
        neg_prompt,
        embellish_prompt,
        style_prompt,
        embedding_model,
        embedding_provider,
        external_api_key="",
        temperature=0.7,
        max_tokens=2048,
        seed=0,
        random=False,
        history_steps=10,
        keep_alive=False,
        top_k=40,
        top_p=0.2,
        repeat_penalty=1.1,
        stop_string=None,
        images=None,
        mode=True,
        clear_history=False,
        text_cleanup=True,
        tool=None,
        tool_input=None,
        prime_directives=None,
        enable_RAG=False,
        query_type="global",
        preset="Default",
        rag_folder_name=None,
        task=None,
        fill_mask=False,
        output_mask_select="",
        precision="fp16",
        attention="sdpa",
        aspect_ratio="16:9",
        top_k_search=3
    ):

        if external_api_key != "":
            llm_api_key = external_api_key
        else:
            llm_api_key = get_api_key(f"{llm_provider.upper()}_API_KEY", llm_provider)

        print(f"LLM API key: {llm_api_key[:5]}...")
        if prime_directives is not None:
            system_message_str = prime_directives
        else:
            system_message = self.assistants.get(assistant, "")
            system_message_str = json.dumps(system_message)

        # Validate LLM model
        validate_models(llm_model, llm_provider, "LLM", base_ip, port, llm_api_key)

        # Validate embedding model
        validate_models(embedding_model, embedding_provider, "embedding", base_ip, port, llm_api_key)

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
        # Handle tools
        try:
            if tool and tool != "None":
                selected_tool = self.agent_tools.get(tool)
                if not selected_tool:
                    raise ValueError(f"Invalid agent tool selected: {tool}")

                # Prepare tool execution message
                tool_message = f"Execute the {tool} tool with the following input: {prompt}"
                system_prompt = json.dumps(selected_tool.system_prompt)

                # Send request to LLM for tool execution
                generated_text =await send_request(
                    llm_provider=llm_provider,
                    base_ip=base_ip,
                    port=port,
                    images=images,
                    model=llm_model,
                    system_message=system_prompt,
                    user_message=tool_message,
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
                )
                # Parse the generated text for function calls
                function_call = None
                try:
                    response_data = json.loads(generated_text)
                    if 'function_call' in response_data:
                        function_call = response_data['function_call']
                        generated_text = response_data['content']
                except json.JSONDecodeError:
                    pass  # The response wasn't JSON, so it's just the generated text

                # Execute the tool with the LLM's response
                tool_args = {
                    "input": prompt,
                    "llm_response": generated_text,
                    "function_call": function_call,
                    "omni_input": tool_input,
                    "name": selected_tool.name,
                    "description": selected_tool.description,
                    "system_prompt": selected_tool.system_prompt
                }
                tool_result = selected_tool.execute(tool_args)

                # Update messages
                messages.append({"role": "user", "content": prompt})
                messages.append({
                    "role": "assistant",
                    "content": json.dumps(tool_result) if isinstance(tool_result, dict) else str(tool_result)
                })

                # Process the tool output
                if isinstance(tool_result, dict):
                    if "error" in tool_result:
                        generated_text = f"Error in {tool}: {tool_result['error']}"
                        tool_output = None
                    elif selected_tool.output_type and selected_tool.output_type in tool_result:
                        tool_output = tool_result[selected_tool.output_type]
                        generated_text = f"Agent {tool} executed successfully. Output generated."
                    else:
                        tool_output = tool_result
                        generated_text = str(tool_output)
                else:
                    tool_output = tool_result
                    generated_text = str(tool_output)

                return {
                    "Question": prompt,
                    "Response": generated_text,
                    "Negative": self.neg_prompts.get(neg_prompt, ""),
                    "Tool_Output": tool_output,
                    "Retrieved_Image": None  # No image retrieved in tool execution
                }
            else:
                response = await self.generate_response(
                    enable_RAG,
                    query_type,
                    prompt,
                    preset,
                    llm_provider,
                    base_ip,
                    port,
                    images,
                    llm_model,
                    system_message_str,
                    messages,
                    temperature,
                    max_tokens,
                    random,
                    top_k,
                    top_p,
                    repeat_penalty,
                    stop,
                    seed,
                    keep_alive,
                    llm_api_key,
                    task,
                    fill_mask,
                    output_mask_select,
                    precision,
                    attention
                )

                generated_text = response.get("Response")
                selected_neg_prompt_name = neg_prompt 
                omni = response.get("Tool_Output")
                retrieved_image = response.get("Retrieved_Image")  
                retrieved_mask = response.get("Mask")

                
                # Update messages
                messages.append({"role": "user", "content": prompt})
                messages.append({"role": "assistant", "content": generated_text})
                
                text_result = str(generated_text).strip()

                if mode:
                    embellish_content = self.embellish_prompts.get(embellish_prompt, "").strip()
                    style_content = self.style_prompts.get(style_prompt, "").strip()
           
                    lines = [line.strip() for line in text_result.split('\n') if line.strip()]
                    combined_prompts = []
                    
                    for line in lines:
                        if text_cleanup:
                            line = clean_text(line)
                        formatted_line = f"{embellish_content} {line} {style_content}".strip()
                        combined_prompts.append(formatted_line)
                    
                    combined_prompt = "\n".join(formatted_line for formatted_line in combined_prompts)
                    # Handle negative prompts
                    if selected_neg_prompt_name == "AI_Fill":
                        try:
                            neg_system_message = self.assistants.get("NegativePromptEngineer")
                            if not neg_system_message:
                                logger.error("NegativePromptEngineer not found in assistants configuration")
                                negative_prompt = "Error: NegativePromptEngineer not configured"
                            else:
                                user_message = f"Generate negative prompts for the following prompt:\n{text_result}"
                                
                                system_message_str = json.dumps(neg_system_message)
                                
                                logger.info(f"Requesting negative prompts for prompt: {text_result[:100]}...")
                                
                                neg_response = await send_request(
                                    llm_provider=llm_provider,
                                    base_ip=base_ip,
                                    port=port,
                                    images=None, 
                                    llm_model=llm_model,
                                    system_message=system_message_str,
                                    user_message=user_message,
                                    messages=[],  # Fresh context for negative generation
                                    seed=seed,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    random=random,
                                    top_k=top_k,
                                    top_p=top_p,
                                    repeat_penalty=repeat_penalty,
                                    stop=stop,
                                    keep_alive=keep_alive,
                                    llm_api_key=llm_api_key
                                )
                                
                                logger.debug(f"Received negative prompt response: {neg_response}")
                                
                                if neg_response:
                                    negative_lines = []
                                    for line in neg_response.split('\n'):
                                        line = line.strip()
                                        if line:
                                            negative_lines.append(line)
                                    
                                    while len(negative_lines) < len(lines):
                                        negative_lines.append(negative_lines[-1] if negative_lines else "")
                                    negative_lines = negative_lines[:len(lines)]
                                    
                                    negative_prompt = "\n".join(negative_lines)
                                else:
                                    negative_prompt = "Error: Empty response from LLM"
                        except Exception as e:
                            logger.error(f"Error generating negative prompts: {str(e)}", exc_info=True)
                            negative_prompt = f"Error generating negative prompts: {str(e)}"
                        
                    elif neg_prompt != "None":
                        neg_content = self.neg_prompts.get(neg_prompt, "").strip()
                        negative_lines = [neg_content for _ in range(len(lines))]
                        negative_prompt = "\n".join(negative_lines)
                    else:
                        negative_prompt = ""  

                else:
                    combined_prompt = text_result
                    negative_prompt = ""

                try:
                    if isinstance(retrieved_image, torch.Tensor):
                        # Ensure it's in the correct format (B, C, H, W)
                        if retrieved_image.dim() == 3:  # Single image (C, H, W)
                            image_tensor = retrieved_image.unsqueeze(0)  # Add batch dimension
                        else:
                            image_tensor = retrieved_image  # Already batched

                        # Create matching batch masks
                        batch_size = image_tensor.shape[0]
                        height = image_tensor.shape[2]
                        width = image_tensor.shape[3]
                        
                        # Create white masks (all ones) for each image in batch
                        mask_tensor = torch.ones((batch_size, 1, height, width), 
                                              dtype=torch.float32, 
                                              device=image_tensor.device)
                        
                        if retrieved_mask is not None:
                            # If we have masks, process them to match the batch
                            if isinstance(retrieved_mask, torch.Tensor):
                                if retrieved_mask.dim() == 3:  # Single mask
                                    mask_tensor = retrieved_mask.unsqueeze(0)
                                else:
                                    mask_tensor = retrieved_mask
                            else:
                                # Process retrieved_mask if it's not a tensor
                                mask_tensor = process_mask(retrieved_mask, image_tensor)
                    else:
                        image_tensor, default_mask_tensor = process_images_for_comfy(
                            retrieved_image, 
                            self.placeholder_image_path
                        )
                        mask_tensor = default_mask_tensor

                        if retrieved_mask is not None:
                            mask_tensor = process_mask(retrieved_mask, image_tensor)
                    return (
                        prompt,
                        combined_prompt,
                        negative_prompt,
                        omni,
                        image_tensor,
                        mask_tensor,
                    )

                except Exception as e:
                    logger.error(f"Exception in image processing: {str(e)}", exc_info=True)
                    placeholder_image, placeholder_mask = load_placeholder_image(self.placeholder_image_path)
                    return (
                        prompt,
                        f"Error: {str(e)}",
                        "",
                        None,
                        placeholder_image,
                        placeholder_mask
                    )

        except Exception as e:
            logger.error(f"Exception occurred in process_chat: {str(e)}", exc_info=True)
            placeholder_image, placeholder_mask = load_placeholder_image(self.placeholder_image_path)
            return (
                prompt,
                f"Error: {str(e)}",
                "",
                None,
                placeholder_image,
                placeholder_mask
            )

    async def generate_response(
        self,
        enable_RAG,
        query_type,
        prompt,
        preset,
        llm_provider,
        base_ip,
        port,
        images,
        llm_model,
        system_message_str,
        messages,
        temperature,
        max_tokens,
        random,
        top_k,
        top_p,
        repeat_penalty,
        stop,
        seed,
        keep_alive,
        llm_api_key,
        task=None,
        fill_mask=False,
        output_mask_select="",
        precision="fp16",
        attention="sdpa",
    ):
        response_strategies = {
            "graphrag": self.graphrag_app.query,
            "colpali": self.colpali_app.query,
            "florence": self.florence_app.run_florence,
            "normal": lambda: send_request(
                llm_provider=llm_provider,
                base_ip=base_ip,
                port=port,
                images=images,
                llm_model=llm_model,
                system_message=system_message_str,
                user_message=prompt,
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
                attention=attention
            ),
        }
        florence_tasks = list(self.florence_prompts.keys())
        if enable_RAG:
            if query_type == "colpali" or query_type == "colpali-v1.2" or query_type == "colqwen2":
                strategy = "colpali"
            else:  # For "global", "local", and "naive" query types
                strategy = "graphrag"
        elif task and task.lower() != 'none' and task in florence_tasks:
            strategy = "florence"
        else:
            strategy = "normal"

        print(f"Strategy: {strategy}")

        try:
            if strategy == "colpali":
                # Ensure the model is loaded before querying
                if self.cached_colpali_model is None:
                    self.cached_colpali_model = self.colpali_app.get_colpali_model(query_type)
                response = await response_strategies[strategy](prompt=prompt, query_type=query_type, system_message_str=system_message_str)
                return response
            elif strategy == "graphrag":
                response = await response_strategies[strategy](prompt=prompt, query_type=query_type, preset=preset) 
                return {
                        "Question": prompt,
                        "Response": response[0],
                        "Negative": "",
                        "Tool_Output": response[1],
                        "Retrieved_Image": None,
                        "Mask": None
                    }
            elif strategy == "florence":
                task_content = self.florence_prompts.get(task, "")
                response = await response_strategies[strategy](
                    images=images,
                    task=task,
                    task_prompt=task_content,
                    llm_model=llm_model,
                    precision=precision,
                    attention=attention,
                    fill_mask=fill_mask,
                    output_mask_select=output_mask_select,
                    keep_alive=keep_alive,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repeat_penalty,
                    seed=seed,
                    text_input=prompt,
                )
                print("Florence response:", response)
                return response
            else:
                response = await response_strategies[strategy]()
                print("Normal response:", response)
                return {
                    "Question": prompt,
                    "Response": response,
                    "Negative": "",
                    "Tool_Output": None,
                    "Retrieved_Image": None,
                    "Mask": None
                }

        except Exception as e:
            logger.error(f"Error processing strategy: {strategy}")
            return {
                "Question": prompt,
                "Response": f"Error processing task: {str(e)}",
                "Negative": "",
                "Tool_Output": {"error": str(e)},
                "Retrieved_Image": None,
                "Mask": None
            }

    def process_chat_wrapper(self, *args, **kwargs):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        logger.debug(f"process_chat_wrapper kwargs: {kwargs}")
        logger.debug(f"External LLM API Key: {kwargs.get('external_api_key', 'Not provided')}")
        return loop.run_until_complete(self.process_chat(*args, **kwargs))

    @classmethod
    def INPUT_TYPES(cls):
        node = cls()
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "The main text input for the chat or query."}),
                "llm_provider": (["xai","llamacpp", "ollama", "kobold", "lmstudio", "textgen", "groq", "gemini", "openai", "anthropic", "mistral", "transformers"], {"default": node.llm_provider, "tooltip": "The provider of the language model to be used."}),
                "llm_model": ((), {"tooltip": "The specific language model to be used for processing."}),
                "base_ip": ("STRING", {"default": node.base_ip, "tooltip": "IP address of the LLM server."}),
                "port": ("STRING", {"default": node.port, "tooltip": "Port number for the LLM server connection."}),               
            },
            "optional": {
                "images": ("IMAGE", {"list": True, "tooltip": "Input image(s) for visual processing or context."}),
                "precision": (['fp16','bf16','fp32','int8','int4'],{"default": 'bf16', "tooltip": "Select preccision on Transformer models."}),
                "attention": (['flash_attention_2','sdpa','xformers', 'Shrek_COT_o1'],{"default": 'sdpa', "tooltip": "Select attention mechanism on Transformer models."}),
                "assistant": ([name for name in node.assistants.keys()], {"default": node.assistant, "tooltip": "The pre-defined assistant personality to use for responses."}),
                "tool": (["None"] + [name for name in node.agent_tools.keys()], {"default": "None", "tooltip": "Selects a specific tool or agent for task execution."}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1, "tooltip": "Controls randomness in output generation. Higher values increase creativity but may reduce coherence."}),
                "max_tokens": ("INT", {"default": 2048, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Maximum number of tokens to generate in the response."}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100, "tooltip": "Limits the next token selection to the K most likely tokens."}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "Cumulative probability cutoff for token selection."}),
                "repeat_penalty": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Penalizes repetition in generated text."}),
                "stop_string": ([name for name in node.stop_strings.keys()], {"tooltip": "Specifies a string at which text generation should stop."}),
                "seed": ("INT", {"default": 94687328150, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed for reproducible outputs."}),
                "random": ("BOOLEAN", {"default": False, "label_on": "Seed", "label_off": "Temperature", "tooltip": "Toggles between using a fixed seed or temperature-based randomness."}),
                "history_steps": ("INT", {"default": 10, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Number of previous conversation turns to consider for context."}),
                "clear_history": ("BOOLEAN", {"default": False, "label_on": "Clear History", "label_off": "Keep History", "tooltip": "Option to clear or retain conversation history."}),
                "keep_alive": ("BOOLEAN", {"default": False, "label_on": "Keeps Model on Memory", "label_off": "Unloads Model from Memory", "tooltip": "Determines whether to keep the model loaded in memory between calls."}),
                "text_cleanup": ("BOOLEAN", {"default": True, "label_on": "Clean Response", "label_off": "Raw Text", "tooltip": "Applies text cleaning to the generated output."}),
                "mode": ("BOOLEAN", {"default": False, "label_on": "Using SD Mode", "label_off": "Using Chat Mode", "tooltip": "Switches between Stable Diffusion prompt generation and standard chat mode."}),
                "embellish_prompt": ([name for name in node.embellish_prompts.keys()], {"tooltip": "Adds pre-defined embellishments to the prompt."}),
                "style_prompt": ([name for name in node.style_prompts.keys()], {"tooltip": "Applies a pre-defined style to the prompt."}),
                "neg_prompt": ([name for name in node.neg_prompts.keys()], {"tooltip": "Adds a negative prompt to guide what should be avoided in generation."}),              
                "fill_mask": ("BOOLEAN", {"default": False, "label_on": "Fill Mask", "label_off": "No Fill", "tooltip": "Option to fill masks for Florence tasks."}),
                "output_mask_select": ("STRING", {"default": ""}),
                "task": ([name for name in node.florence_prompts.keys()], {"default": "None", "tooltip": "Select a Florence task."}),
                "embedding_provider": (["llamacpp", "ollama", "kobold", "lmstudio", "textgen", "groq", "gemini", "openai", "anthropic", "mistral", "sentence_transformers"], {"default": node.embedding_provider, "tooltip": "Provider for text embedding model."}),
                "embedding_model": ((), {"tooltip": "Specific embedding model to use."}),
                "tool_input": ("OMNI", {"default": None, "tooltip": "Additional input for the selected tool."}),
                "prime_directives": ("STRING", {"forceInput": True, "tooltip": "System message or prime directive for the AI assistant."}),
                "external_api_key":("STRING", {"default": "", "tooltip": "If this is not empty, it will be used instead of the API key from the .env file. Make sure it is empty to use the .env file."}),
                "top_k_search": ("INT", {"default": 3, "min": 1, "max": 10, "tooltip": "Find top scored image(s) from RAG."}),
                "aspect_ratio": (["1:1", "9:16", "16:9"], {"default": "16:9", "tooltip": "Select the aspect ratio for the image."}),
                "enable_RAG": ("BOOLEAN", {"default": False, "label_on": "RAG is Enabled", "label_off": "RAG is Disabled", "tooltip": "Enables Retrieval-Augmented Generation for enhanced context."}),
                "query_type": (["global", "local", "naive", "colpali", "colqwen2", "colpali-v1.2"], {"default": "global", "tooltip": "Selects the type of query strategy for RAG."}),
                "preset": (["Default", "Detailed", "Quick", "Bullet", "Comprehensive", "High-Level", "Focused"], {"default": "Default"}),
            },
            "hidden": {
                "model": ("STRING", {"default": ""}),
                "rag_root_dir": ("STRING", {"default": "rag_data"})
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        node = cls()

        llm_provider = kwargs.get('llm_provider', node.llm_provider)
        embedding_provider = kwargs.get('embedding_provider', node.embedding_provider)
        base_ip = kwargs.get('base_ip', node.base_ip)
        port = kwargs.get('port', node.port)
        query_type = kwargs.get('query_type', node.query_type)
        external_api_key = kwargs.get('external_api_key', '')
        task = kwargs.get('task', node.task)

        # Determine which API key to use
        def get_api_key_with_fallback(provider, external_api_key):
            if external_api_key and external_api_key != '':
                return external_api_key
            try:
                # print(f"Using {provider} API key from .env file")
                api_key = get_api_key(f"{provider.upper()}_API_KEY", provider)
                # print(f" {api_key} API key for {provider} found in .env file")
                return api_key

            except ValueError:
                return None

        api_key = get_api_key_with_fallback(llm_provider, external_api_key)

        # Check for changes
        llm_provider_changed = llm_provider != node.llm_provider
        embedding_provider_changed = embedding_provider != node.embedding_provider
        api_key_changed = external_api_key != node.external_api_key
        base_ip_changed = base_ip != node.base_ip
        port_changed = port != node.port
        query_type_changed = query_type != node.query_type
        task_changed = task != node.task

        # Always fetch new models if the provider, API key, base_ip, or port has changed
        if llm_provider_changed or api_key_changed or base_ip_changed or port_changed:
            try:
                new_llm_models = get_models(llm_provider, base_ip, port, api_key)
            except Exception as e:
                print(f"Error fetching LLM models: {e}")
                new_llm_models = []
            llm_model_changed = new_llm_models != node.llm_model
        else:
            llm_model_changed = False

        if embedding_provider_changed or api_key_changed or base_ip_changed or port_changed:
            try:
                new_embedding_models = get_models(embedding_provider, base_ip, port, api_key)
            except Exception as e:
                print(f"Error fetching embedding models: {e}")
                new_embedding_models = []
            embedding_model_changed = new_embedding_models != node.embedding_model
        else:
            embedding_model_changed = False

        if (llm_provider_changed or embedding_provider_changed or llm_model_changed or 
            embedding_model_changed or query_type_changed or task_changed or api_key_changed or
            base_ip_changed or port_changed):

            node.llm_provider = llm_provider
            node.embedding_provider = embedding_provider
            node.base_ip = base_ip
            node.port = port
            node.external_api_key = external_api_key
            node.query_type = query_type
            node.task = task

            if llm_model_changed:
                node.llm_model = new_llm_models
            if embedding_model_changed:
                node.embedding_model = new_embedding_models

            # Update other attributes
            for attr in ['seed', 'random', 'history_steps', 'clear_history', 'mode', 
                        'keep_alive', 'tool', 'enable_RAG', 'preset']:
                setattr(node, attr, kwargs.get(attr, getattr(node, attr)))

            return True

        return False

    RETURN_TYPES = ("STRING", "STRING", "STRING", "OMNI", "IMAGE", "MASK")
    RETURN_NAMES = ("Question", "Response", "Negative", "Tool_Output", "Retrieved_Image", "Mask")

    OUTPUT_TOOLTIPS = (
        "The original input question or prompt.",
        "The generated response from the language model.",
        "The negative prompt used (if applicable) for guiding image generation.",
        "Output from the selected tool, which can be code or any other data type.",
        "An image retrieved by the RAG system, if applicable.",
        "Mask image generated by Florence tasks."
    )
    FUNCTION = "process_chat_wrapper"
    OUTPUT_NODE = True
    CATEGORY = "ImpactFrames💥🎞️"
    DESCRIPTION = "ComfyUI, Support API and Local LLM providers and RAG capabilities. Processes text prompts, handles image inputs, and integrates with different language models and indexing strategies."

