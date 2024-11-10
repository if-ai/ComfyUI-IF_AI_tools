import os
import re
import sys
import glob
import uuid
import yaml
import time
import json
import queue
import shutil
import asyncio
import logging
import aiohttp
import requests
import importlib
import traceback
import folder_paths
# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import numpy as np
from nano_graphrag.graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs
#from litellm import completion, embedding, text_completion

from .send_request import create_embedding, send_request

# Set LiteLLM to be verbose
#from litellm import set_verbose
set_verbose = True

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

from .utils import get_api_key



from .graph_visualize_tool import visualize_graph

class GraphRAGapp:
    def __init__(self):
        self.comfy_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.rag_dir = os.path.join(folder_paths.base_path, "custom_nodes",  "ComfyUI_IF_AI_tools", "IF_AI", "rag")
        self._rag_root_dir = None
        self._input_dir = None
        self.embedding_func = None
        self.graphrag = None

    @property
    def rag_root_dir(self):
        return self._rag_root_dir

    @rag_root_dir.setter
    def rag_root_dir(self, value):
        self._rag_root_dir = value
        self._input_dir = os.path.join(value, "input") if value else None
        logger.debug(f"rag_root_dir setter: set to {self._rag_root_dir}")
        logger.debug(f"input_dir set to {self._input_dir}")
        if self._input_dir:
            os.makedirs(self._input_dir, exist_ok=True)
            logger.debug(f"Created input directory: {self._input_dir}")

    def set_rag_root_dir(self, rag_folder_name):
        if rag_folder_name:
            new_rag_root_dir = os.path.join(self.rag_dir, rag_folder_name)
        else:
            new_rag_root_dir = os.path.join(self.rag_dir, "rag_data")

        self._rag_root_dir = new_rag_root_dir
        self._input_dir = os.path.join(new_rag_root_dir, "input")
        
        # Ensure directories exist
        os.makedirs(self._rag_root_dir, exist_ok=True)
        os.makedirs(self._input_dir, exist_ok=True)

        logger.debug(f"set_rag_root_dir: rag_root_dir set to {self._rag_root_dir}")
        logger.debug(f"set_rag_root_dir: input_dir set to {self._input_dir}")
        return self._rag_root_dir

    def _save_settings_to_path(self, settings_path):
        """Save settings to a specific path, overwriting if it already exists."""
        try:
            with open(settings_path, 'w') as f:
                yaml.dump(
                    self.settings,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                    default_style=None,
                    Dumper=yaml.SafeDumper,
                )
            logger.info(f"Settings saved to {settings_path}")
        except Exception as e:
            logger.error(f"Error saving settings to {settings_path}: {str(e)}")

    def save_settings(self):
        """Save settings to both the RAG-specific folder and the main RAG directory."""
        if self.settings_path:
            self._save_settings_to_path(self.settings_path)
        else:
            logger.warning("RAG-specific settings path not set. Unable to save settings to RAG folder.")

        # Save a copy to the main RAG directory
        rag_dir_settings_path = os.path.join(self.rag_dir, 'settings.yaml')
        self._save_settings_to_path(rag_dir_settings_path)

        return self.settings

    async def setup_and_initialize_folder(self, rag_folder_name, settings):
        try:
            rag_root = os.path.join(self.rag_dir, rag_folder_name)
            logger.debug(f"rag_root set to: {rag_root}")
            self.settings_path = os.path.join(rag_root, 'settings.yaml')

            self._rag_root_dir = rag_root
            self._input_dir = os.path.join(rag_root, "input")

            os.makedirs(rag_root, exist_ok=True)
            logger.info(f"Created/ensured folder: {rag_root}")

            # Create the input directory
            os.makedirs(self._input_dir, exist_ok=True)
            logger.info(f"Created/ensured input directory: {self._input_dir}")

            # Update settings.yaml with UI settings
            self.settings = self._create_settings_from_ui(settings)
            self.save_settings()

            # Add a short delay to ensure settings are saved
            await asyncio.sleep(1)

            # Create the GraphRAG instance here
            await self.setup_embedding_func()
            self.graphrag = GraphRAG(
                working_dir=self._rag_root_dir,
                enable_llm_cache=True,
                best_model_func=self.unified_model_if_cache,
                cheap_model_func=self.unified_model_if_cache,
                embedding_func=self.embedding_func,
            )

            result = {
                "status": "success",
                "message": f"Folder initialized: {rag_root}",
                "rag_root_dir": rag_root,
            }
            logger.debug(f"Final result: {result}")
            logger.debug(f"self.rag_root_dir after initialization: {self.rag_root_dir}")
            return result

        except Exception as e:
            logger.error(f"Error in setup_and_initialize_folder: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _create_settings_from_ui(self, ui_settings):
        """
        Create settings.yaml from UI settings with proper type conversion.
        """
        settings = {
            'embedding_provider': str(ui_settings.get('embedding_provider', 'sentence_transformers')),
            'embedding_model': str(ui_settings.get('embedding_model', 'avsolatorio/GIST-small-Embedding-v0')),
            'base_ip': str(ui_settings.get('base_ip', 'localhost')),
            'port': str(ui_settings.get('port', '11434')),
            'llm_provider': str(ui_settings.get('llm_provider', 'ollama')),
            'llm_model': str(ui_settings.get('llm_model', 'llama3.1:latest')),
            'temperature': float(ui_settings.get('temperature', '0.7')),
            'max_tokens': int(ui_settings.get('max_tokens', '2048')),
            'stop': None if ui_settings.get('stop', 'None') == 'None' else str(ui_settings.get('stop')),
            'keep_alive': ui_settings.get('keep_alive', 'False').lower() == 'true',  # Convert to boolean
            'top_k': int(ui_settings.get('top_k', '40')),
            'top_p': float(ui_settings.get('top_p', '0.90')),
            'repeat_penalty': float(ui_settings.get('repeat_penalty', '1.2')),
            'seed': None if ui_settings.get('seed', 'None') == 'None' else int(ui_settings.get('seed')),
            'rag_folder_name': str(ui_settings.get('rag_folder_name', 'rag_data')),
            'query_type': str(ui_settings.get('query_type', 'global')),
            'community_level': int(ui_settings.get('community_level', '2')),
            'preset': str(ui_settings.get('preset', 'Default')),
            'external_llm_api_key': str(ui_settings.get('external_llm_api_key', '')),
            'random': ui_settings.get('random', 'False').lower() == 'true',  # Convert to boolean
            'prime_directives': None if ui_settings.get('prime_directives', 'None') == 'None' else str(ui_settings.get('prime_directives')),
            'prompt': str(ui_settings.get('prompt', 'Who helped Safiro infiltrate the Zaltar Organisation?')),
            'response_format': str(ui_settings.get('response_format', 'json')),
            'precision': str(ui_settings.get('precision', 'fp16')),
            'attention': str(ui_settings.get('attention', 'sdpa')),
            'aspect_ratio': str(ui_settings.get('aspect_ratio', '16:9')),
            'top_k_search': int(ui_settings.get('top_k_search', '3')),
        }
        return settings

    def load_settings(self):
        if self._rag_root_dir:
            self.settings_path = os.path.join(self._rag_root_dir, "settings.yaml")
        else:
            self.settings_path = os.path.join(self.rag_dir, "settings.yaml")

        if os.path.exists(self.settings_path):
            with open(self.settings_path, 'r') as f:
                try:
                    self.settings = yaml.safe_load(f)
                    logger.info(f"Loaded settings from {self.settings_path}")
                except yaml.YAMLError as e:
                    logger.error(f"Error parsing settings file: {str(e)}")
                    self.settings = {}
        else:
            logger.warning(f"Settings file not found at {self.settings_path}")
            self.settings = {}

        return self.settings

    async def setup_embedding_func(self, **kwargs) -> None:
        settings = self.load_settings()
        base_ip = kwargs.pop("base_ip", settings.get("base_ip", "localhost"))
        port = kwargs.pop("port", settings.get("port", "11434"))
        #base64_image = kwargs.pop("base64_image", settings.get("base64_image", None))
        embedding_provider = settings.get('embedding_provider', 'sentence_transformers')
        embedding_model = settings.get('embedding_model', 'avsolatorio/GIST-small-Embedding-v0')
       
        embedding_api_key = settings.get('external_llm_api_key') if settings.get('external_llm_api_key') != "" else get_api_key(f"{embedding_provider.upper()}_API_KEY", embedding_provider)
        
        api_base = f"http://{base_ip}:{port}" if embedding_provider in ["ollama", "lmstudio", "llamacpp", "textgen"] else f"https://api.{embedding_provider}.com"
        
        if embedding_provider in ["openai", "mistral", "lmstudio", "llamacpp", "textgen", "ollama"]:
            embedding_dim = 1536 if embedding_provider in ["openai", "mistral"] else 768
            @wrap_embedding_func_with_attrs(embedding_dim=embedding_dim, max_token_size=8192)
            async def embedding_func(texts: list[str]) -> np.ndarray:
                embeddings = [] # Initialize embeddings as a list

                for text in texts: # Iterate through each text in the input list
                    embedding = await create_embedding(
                        embedding_provider, api_base, embedding_model, [text], embedding_api_key # Send single text at a time
                    )
                    if embedding is None:
                        raise ValueError(
                            f"Failed to generate embeddings with {embedding_provider}/{embedding_model}"
                        )
                    embeddings.append(embedding) # Append individual embedding to list

                return np.array(embeddings) # Convert list of embeddings to NumPy array
        
        elif embedding_provider == "sentence_transformers":
            from sentence_transformers import SentenceTransformer
            EMBED_MODEL = SentenceTransformer(embedding_model)
            embedding_dim = EMBED_MODEL.get_sentence_embedding_dimension()
            max_token_size = EMBED_MODEL.max_seq_length

            @wrap_embedding_func_with_attrs(
                embedding_dim=embedding_dim, max_token_size=max_token_size
            )
            async def embedding_func(texts: list[str]) -> np.ndarray:
                return EMBED_MODEL.encode(texts, normalize_embeddings=True)

        self.embedding_func = embedding_func

    def remove_if_exist(self, file):
        if os.path.exists(file):
            os.remove(file)


    async def unified_model_if_cache(self, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
        settings = self.load_settings()
        logger.info(f"Loaded settings for LLM: {settings}")
        base_ip = kwargs.pop("base_ip", settings.get("base_ip", "localhost"))
        port = kwargs.pop("port", settings.get("port", "11434"))
        llm_provider = kwargs.pop("llm_provider", settings.get("llm_provider", "ollama"))
        llm_model = kwargs.pop("llm_model", settings.get("llm_model", "llama3.2:latest"))
        temperature = float(kwargs.pop("temperature", settings.get("temperature", "0.7")))
        max_tokens = int(kwargs.pop("max_tokens", settings.get("max_tokens", "2048")))
        keep_alive = kwargs.pop("keep_alive", settings.get("keep_alive", "False"))
        top_k = int(kwargs.pop("top_k", settings.get("top_k", "50")))
        top_p = float(kwargs.pop("top_p", settings.get("top_p", "0.95")))
        presence_penalty = float(kwargs.pop("repeat_penalty", settings.get("repeat_penalty", "1.2")))
        llm_api_key = settings.get('external_llm_api_key') if settings.get('external_llm_api_key') != "" else get_api_key(f"{settings['llm_provider'].upper()}_API_KEY", settings['llm_provider'])
        seed = kwargs.pop("seed", settings.get("seed", "None"))
        random = kwargs.pop("random", settings.get("random", "False"))
        response_format = kwargs.pop("response_format", settings.get("response_format", "json"))       
        stop = kwargs.pop("stop", settings.get("stop", "None"))
        if stop is None or stop.lower() == "none":
            stop = None

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        
        if hashing_kv is not None:
            args_hash = compute_args_hash(llm_model, messages)
            if_cache_return = await hashing_kv.get_by_id(args_hash)
            if if_cache_return is not None:
                return if_cache_return["return"]
        
        print(f"Prompt: {prompt}")            
        user_message = prompt
        try:
            response = await send_request(
                llm_provider=llm_provider,
                base_ip=base_ip,
                port=port,
                images=None, 
                llm_model=llm_model,
                system_message=system_prompt,
                user_message=user_message,
                messages=messages,
                llm_api_key=llm_api_key,
                seed=seed,
                random=random,
                stop=stop,
                keep_alive=keep_alive,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                repeat_penalty=presence_penalty,
                tools=None,
                tool_choice=None
            )
            
            # Handle different response formats
            if isinstance(response, dict) and "message" in response:
                result = response["message"]["content"]
            elif isinstance(response, dict) and "content" in response:
                result = response["content"]
            elif isinstance(response, str):
                result = response
            else:
                raise ValueError(f"Unexpected response format: {type(response)}")
                
            if hashing_kv is not None:
                await hashing_kv.upsert({args_hash: {"return": result, "model": llm_model}})
            return result
            
        except Exception as e:
            logger.error(f"Error during LLM completion: {str(e)}")
            logger.error(f"Response type: {type(response)}")
            logger.error(f"Response content: {response}")
            raise ValueError(f"Error during LLM completion: {str(e)}")
    
    def get_preset_values(self, preset, kwargs, settings):
        preset_values = {
            "Default": ("2", "Multiple Paragraphs"),
            "Detailed": ("4", "Multi-Page Report"),
            "Quick": ("1", "Single Paragraph"),
            "Bullet": ("2", "List of 3-7 Points"),
            "Comprehensive": ("5", "Multi-Page Report"),
            "High-Level": ("1", "Single Page"),
            "Focused": ("3", "Multiple Paragraphs"),
        }

        if preset.startswith(tuple(preset_values.keys())):
            return preset_values[preset.split()[0]]
        elif preset == "Custom Query":
            return (
                kwargs.pop("community_level", settings.get("community_level", "2")),
                kwargs.pop("response_type", settings.get("response_type", "Multiple Paragraphs"))
            )
        else:
            return ("2", "Multiple Paragraphs")
    
    async def query(self, prompt, query_type, preset):
        logger.debug(f"Query - GraphRAG instance id: {id(self.graphrag)}")
        logger.debug(f"Query - Working directory: {self._rag_root_dir}")
        
        settings = self.load_settings()
        working_dir = os.path.join(self.rag_dir, settings.get("rag_folder_name"))
        print(f"Working directory: {working_dir}")
        
        if self.graphrag is None:
            logger.info("GraphRAG instance not initialized. Initializing...")
            await self.setup_embedding_func()
            self.graphrag = GraphRAG(
                working_dir=working_dir,
                enable_llm_cache=True,
                best_model_func=self.unified_model_if_cache,
                cheap_model_func=self.unified_model_if_cache,
                embedding_func=self.embedding_func,
            )
        
        community_level, response_type = self.get_preset_values(preset, {}, settings)
        print(f"Community level: {community_level}, Response type: {response_type}")
        
        for filename in ["vdb_entities.json", "kv_store_full_docs.json", "kv_store_text_chunks.json"]:
            file_path = os.path.join(working_dir, filename)
            if os.path.exists(file_path):
                logger.debug(f"File exists: {file_path}, size: {os.path.getsize(file_path)} bytes")
            else:
                logger.warning(f"File not found: {file_path}")

        try:
            result = await self.graphrag.aquery(
                query=prompt,
                param=QueryParam(
                    mode=query_type,
                    response_type=response_type,
                    level=int(community_level)
                )
            )
            
            # Define the dynamic path for the GraphML file
            graphml_path = os.path.join(self.graphrag.working_dir, "graph_chunk_entity_relation.graphml")
            
            # Call the visualize_graph function to visualize the graph
            try:
                visualize_graph(graphml_path)
            except Exception as viz_error:
                logger.error(f"Error visualizing graph: {str(viz_error)}")
                print(f"Error visualizing graph: {str(viz_error)}")
            
            
            return result, graphml_path
        except Exception as e:
            logger.error(f"Error in GraphRAGapp.query: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error during query: {str(e)}"

    async def insert(self):
        logger.debug("Starting insert function...")
        logger.debug(f"Insert - rag_dir: {self.rag_dir}")
        logger.debug(f"Insert - _rag_root_dir: {self._rag_root_dir}")
        logger.debug(f"Insert - _input_dir: {self._input_dir}")
        settings = self.load_settings()
        print(f"Settings: {settings}")
        
        working_dir = self._rag_root_dir
        insert_input_dir = self._input_dir
        
        print(f"Working directory: {working_dir}")
        print(f"Insert input directory: {insert_input_dir}")
        try:
            logger.debug(f"Listing files in {insert_input_dir}")
            all_texts = []
            for filename in os.listdir(insert_input_dir):
                if filename.endswith(".txt"):
                    file_path = os.path.join(insert_input_dir, filename)
                    logger.debug(f"Reading file: {file_path}")
                    with open(file_path, encoding="utf-8-sig") as f:
                        all_texts.append(f.read())
            
            if not all_texts:
                logger.warning("No text files found in the input directory.")
                return False
            
            combined_text = "\n".join(all_texts)
            logger.debug(f"Combined text length: {len(combined_text)}")

            # Remove existing files
            logger.debug("Removing existing files...")
            for filename in ["vdb_entities.json", "kv_store_full_docs.json", "kv_store_text_chunks.json", "kv_store_community_reports.json", "graph_chunk_entity_relation.graphml"]:
                self.remove_if_exist(os.path.join(working_dir, filename))

            logger.debug("Creating GraphRAG instance...")
            
            # Set up the embedding function before creating the GraphRAG instance
            await self.setup_embedding_func()
            
            # Use the existing graphrag instance or create a new one
            if self.graphrag is None:
                self.graphrag = GraphRAG(
                    working_dir=working_dir,
                    enable_llm_cache=True,
                    best_model_func=self.unified_model_if_cache,
                    cheap_model_func=self.unified_model_if_cache,
                    embedding_func=self.embedding_func,
                )

            start = time.time()
            logger.debug("Inserting text...")
            await self.graphrag.ainsert(combined_text)
            logger.debug(f"Indexing completed in {time.time() - start:.2f} seconds")
            print("indexing time:", time.time() - start)

            # Cleanup step
            extra_folder = os.path.join(self.comfy_dir, settings.get("rag_folder_name"))
            if os.path.exists(extra_folder) and os.path.isdir(extra_folder):
                logger.debug(f"Removing extra folder: {extra_folder}")
                shutil.rmtree(extra_folder)

            return True

        except Exception as e:
            logger.error(f"Error during indexing: {str(e)}")
            logger.error(traceback.format_exc())
            return False