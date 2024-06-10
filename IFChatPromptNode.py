import os
import io
import shutil
import torch
import tempfile
import time
import json
import requests
import base64
import textwrap
import pandas as pd
import pathway as pw
import numpy as np
import anthropic
import openai
import asyncio
import threading
import platform
import sys
import aiohttp
import queue

from PIL import Image
from server import PromptServer
from aiohttp import web, ClientSession
from datetime import date
from dotenv import load_dotenv

from .rag_module import clear_contexts, RAG_READY_EVENT
from .text_cleanup import process_text
from .send_request import send_request
import subprocess

RAG_PROCESS = None

#comfy_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
@PromptServer.instance.routes.post("/IF_ChatPrompt/get_models")
async def get_models_endpoint(request):
    data = await request.json()
    engine = data.get("engine")
    base_ip = data.get("base_ip")
    port = data.get("port")

    node = IFChatPrompt()
    models = node.get_models(engine, base_ip, port)
    return web.json_response(models)

    # New endpoint to save config


class IFChatPrompt:

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("Question", "Response", "Negative",)
    FUNCTION = "process_chat"
    OUTPUT_NODE = True
    CATEGORY = "ImpactFrames💥🎞️"

    @classmethod
    def INPUT_TYPES(cls):
        node = cls()
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "base_ip": ("STRING", {"default": node.base_ip}),
                "port": ("STRING", {"default": node.port}),
                "engine": (["ollama", "kobold", "lms", "textgen", "groq", "openai", "anthropic"], {"default": node.engine}),
                #"model": (node.get_models("node.engine", node.base_ip, node.port), {}),
                "model": ((), {}),
                "assistant": ([name for name in node.assistants.keys()], {"default": node.assistant}),
            },
            "optional": {
                "images": ("IMAGE", ),
                "rag_port": ("STRING", {"default": node.rag_port}),
                "max_tokens": ("INT", {"default": 2048, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100}),
                "top_p": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.1}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 10.0, "step": 0.1}),
                "stop": ("STRING", {"default": "<|end_of_text|>", "multiline": False}),
                "seed": ("INT", {"default": 94687328150, "min": 0, "max": 0xffffffffffffffff}),
                "random": ("BOOLEAN", {"default": False, "label_on": "Seed", "label_off": "Temperature"}),
                "embellish_prompt": ([name for name in node.embellish_prompts.keys()], {}),
                "style_prompt": ([name for name in node.style_prompts.keys()], {}),
                "neg_prompt": ([name for name in node.neg_prompts.keys()], {}),
                "clear_history": ("BOOLEAN", {"default": True, "label_on": "Clear History", "label_off": "Keep History"}),
                "history_steps": ("INT", {"default": 10, "min": 0, "max": 0xffffffffffffffff}),
                "clear_rag": ("BOOLEAN", {"default": False, "label_on": "Clear All", "label_off": "Keep All"}),
                "keep_alive": ("BOOLEAN", {"default": False, "label_on": "Keeps_Model", "label_off": "Unloads_Model"}),
                "text_cleanup": ("BOOLEAN", {"default": True, "label_on": "Apply", "label_off": "Raw Text"}),
                "document": ("STRING", {"multiline": True, "default": "place a comma separated list of file locations here to use as documents"}),
                "mode": ("BOOLEAN", {"default": True, "label_on": "Mode: SD", "label_off": "Mode: Chat"}),
                "reload_rag": ("BOOLEAN", {"default": False, "label_on": "Relaunch", "label_off": "Keep"}),
            },
            "hidden": {
                "model": ("STRING", {"default": ""}),
            },
        }


    @classmethod
    def IS_CHANGED(cls, engine, base_ip, port, assistant, keep_alive, seed, random, history_steps, model):
        node = cls()
        seed_changed = seed != node.seed or random != node.random
        engine_changed = engine != node.engine
        base_ip_changed = base_ip != node.base_ip
        port_changed = port != node.port
        selected_model_changed = node.model != node.get_models(engine, base_ip, port)

        if seed_changed or engine_changed or base_ip_changed or port_changed or selected_model_changed:
            node.engine = engine
            node.base_ip = base_ip
            node.port = port
            node.model = node.get_models(engine, base_ip, port)
            node.assistant = assistant
            node.keep_alive = keep_alive
            node.seed = seed
            node.random = random
            node.history_steps = history_steps
            return True

        return False

    def __init__(self):
        self.base_ip = "localhost"
        self.port = "11434"
        self.rag_port = "8081"
        self.engine = "ollama"
        self.model = ""
        self.assistant = "Cortana"
        self.comfy_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.documents_dir = os.path.join(self.comfy_dir, "input", "IF_AI", "documents")
        self.presets_dir = os.path.join(self.comfy_dir, "input", "IF_AI", "presets")
        self.assistants_file = os.path.join(self.presets_dir, "assistants.json")
        self.assistants = self.load_presets(self.assistants_file)
        self.neg_prompts_file = os.path.join(self.presets_dir, "neg_prompts.json")
        self.embellish_prompts_file = os.path.join(self.presets_dir, "embellishments.json")
        self.style_prompts_file = os.path.join(self.presets_dir, "style_prompts.json")
        self.neg_prompts = self.load_presets(self.neg_prompts_file)
        self.embellish_prompts = self.load_presets(self.embellish_prompts_file)
        self.style_prompts = self.load_presets(self.style_prompts_file)
        self.keep_alive = False
        self.seed = 94687328150
        self.chat_history = []
        self.history_steps = 10
        self.document = ""
        self.chat_history_dir = os.path.join(self.comfy_dir, "output", "IF_AI", "chat_history")
        self.assistant_memory_dir = os.path.join(self.comfy_dir, "output", "IF_AI", "assistant_data")
        self.knowledge_base_dir = os.path.join(self.comfy_dir, "output", "IF_AI", "knowledge_base")
        self.rag_pipeline = None 
        self.reload_rag = False  

    def load_presets(self, file_path):
        with open(file_path, 'r') as f:
            presets = json.load(f)
        return presets

    def get_api_key(self, api_key_name, engine):
        if engine not in ["ollama", "kobold", "lms", "textgen"]:
            # Try to get the key from .env first
            load_dotenv()
            api_key = os.getenv(api_key_name)
            if api_key:
                return api_key
            else:
                # If .env is empty, get the key from os.environ
                api_key = os.getenv(api_key_name) 
                if api_key:
                    return api_key
                else:
                    raise ValueError(f"{api_key_name} not found. Please set it in your .env file or as an environment variable.")
        else:
            print(f'You are using {engine} as the engine, no API key is required.')
            return None

    def get_models(self, engine, base_ip, port):
        if engine == "groq":
            return ["gemma-7b-it", "llama2-70b-4096", "llama3-70b-8192", "llama3-8b-8192","mixtral-8x7b-32768"]

        elif engine == "ollama":
            api_url = f'http://{base_ip}:{port}/api/tags'
            try:
                response = requests.get(api_url)
                response.raise_for_status()
                models = [model['name'] for model in response.json().get('models', [])]
                return models
            except Exception as e:
                print(f"Failed to fetch models from Ollama: {e}")
                return []
        elif engine == "lms":
            api_url = f'http://{base_ip}:{port}/v1/models'
            try:
                response = requests.get(api_url)
                if response.status_code == 200:
                    data = response.json()
                    models = [model['id'] for model in data['data']]
                    return models
                else:
                    print(f"Failed to fetch models from LM Studio. Status code: {response.status_code}")
                    return []
            except requests.exceptions.RequestException as e:
                print(f"Error connecting to LM Studio server: {e}")
                return []
        elif engine == "textgen":
            api_url = f'http://{base_ip}:{port}/v1/internal/model/list'
            try:
                response = requests.get(api_url)
                response.raise_for_status()
                models = response.json()['model_names']
                return models
            except Exception as e:
                print(f"Failed to fetch models from text-generation-webui: {e}")
                return []
        elif engine == "kobold":
            api_url = f'http://{base_ip}:{port}/api/v1/model'
            try:
                response = requests.get(api_url)
                response.raise_for_status()
                model = response.json()['result']
                return [model]
            except Exception as e:
                print(f"Failed to fetch models from Kobold: {e}")
                return []
        elif engine == "anthropic":
            return ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
        elif engine == "openai":
            api_key = self.get_api_key("OPENAI_API_KEY", engine)
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            api_url = "https://api.openai.com/v1/models"
            try:
                response = requests.get(api_url, headers=headers)
                response.raise_for_status()
                models = [model["id"] for model in response.json()["data"]]
                return models
            except Exception as e:
                print(f"Failed to fetch models from OpenAI: {e}")
                return []
        else:
            print(f"Unsupported engine - {engine}")
            return []

    def update_assistant_memory(self, assistant, messages, engine, base_ip, port, base64_image, model,
                             system_message, seed, temperature, max_tokens, random,
                             top_k, top_p, repeat_penalty, stop, keep_alive, api_key):
        """Updates assistant memory with chat history summaries."""
        
        # Check if there are previous chat history summaries to process
        if len(os.listdir(self.assistant_memory_dir)) > 0:
            # Define the prompt for summarization
            summarization_prompt = textwrap.dedent("""
                Your job is to assess a brief chat history and extract any details worth remembering and recording in a knowledge base.
                Your main goal are identifying any relevant data in the chat history related to the user such as preferences, behaviors, etc. and, identifying relevant information, and synthesizing it into concise knowledge pills.
                You are interested in a broad range of information, including but not limited to:
                Facts about people, places, events, objects, concepts, etc.
                Descriptions of processes, instructions, and how to do various tasks
                Opinions, perspectives and analyses on various topics
                Creative ideas, stories, jokes, wordplay, etc.
                When you receive a message, follow these steps:
                Analyze the message for information worth remembering
                If it contains relevant information:
                a. Extract and summarize the key points in 1-2 concise sentences (knowledge pills)
                b. Return the knowledge pills
                If the message does not contain any relevant information, return "No relevant information"
                Focus on brevity and clarity in your knowledge pills. Capture the essence of the information in as few words as possible while maintaining accuracy and context.
                Always identify and catalogue the information in the message as user: and assistant: depending on who made the message,
                think step by step, and then analyze the following set of message history and extract any details worth remembering and recording in a knowledge base:...
                """)

            # Create the directory for the assistant's memory if it doesn't exist
            os.makedirs(self.assistant_memory_dir, exist_ok=True)

            # Generate summary for each message and save as a plain text file
            summary_lines = []
            for message in messages:
                role = message['role']
                content = message['content']
                summary = send_request(engine=engine,
                                    base_ip=base_ip,
                                    port=port,
                                    base64_image=base64_image,
                                    model=model,
                                    system_message=system_message,
                                    user_message=f"{summarization_prompt}\n{role}: {content}",
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
                                    chat_history=[],
                                    api_key=api_key)[0]  # Only return response
                summary_lines.append(summary)

            # Save the summary as a plain text file in the assistant's memory directory
            summary_file = os.path.join(self.assistant_memory_dir, f"{assistant}{date.today().strftime('%Y%m%d')}_summary.txt")
            with open(summary_file, "w") as f:
                f.write("\n".join(summary_lines))
            
        # Save the full chat history as a plain text file
        os.makedirs(self.chat_history_dir, exist_ok=True)
        chat_history_file = os.path.join(self.chat_history_dir, f"{assistant}_chat_history.txt")
        with open(chat_history_file, "w") as f:
            for message in messages:
                f.write(f"{message['role']}: {message['content']}\n")

        # Copy both the chat history and summary files to knowledge_base_dir
        kb_chat_history_file = os.path.join(self.knowledge_base_dir, f"{assistant}_chat_history.txt")
        shutil.copy2(chat_history_file, kb_chat_history_file)
        kb_summary_file = os.path.join(self.knowledge_base_dir, f"{assistant}{date.today().strftime('%Y%m%d')}_summary.txt")
        shutil.copy2(summary_file, kb_summary_file)  # Copy the summary file

    def update_knowledge_base(self, document):
        """Copies new documents from documents_dir to knowledge_base_dir."""
        if document != self.document:
            document_files = [file.strip() for file in document.split(",")]
            for file_path in document_files:
                if os.path.isfile(file_path):
                    file_name = os.path.basename(file_path)
                    # Copy the file to documents_dir
                    dest_path = os.path.join(self.documents_dir, file_name)
                    shutil.copy2(file_path, dest_path)
                    print(f"Copied {file_path} to {dest_path}")
                    os.makedirs(self.knowledge_base_dir, exist_ok=True)
                    # Copy the file from documents_dir to knowledge_base_dir 
                    kb_path = os.path.join(self.knowledge_base_dir, file_name)
                    shutil.copy2(dest_path, kb_path) # Always copy to knowledge base
                    print(f"Copied {dest_path} to {kb_path}")
                else:
                    print(f"File not found: {file_path}")
            self.document = document

    def prepare_messages(self, prompt, assistant, images=None):
        assistant_content = self.assistants.get(assistant, "")
        image_message = textwrap.dedent("""
                Analyze the images provided, search for relevant details from the images to include on your response.
                Reply to the user's specific question or prompt and include relevant details extracted from the images.
            """)
        if images is not None:
            system_message = f"{assistant_content}\n{image_message}"

        else:
            system_message = f"{assistant_content}"


        user_message = prompt if prompt.strip() != "" else "Please provide a general description of the images."

        messages = []

        # Add the conversation history and user message regardless of history being empty
        for message in self.chat_history:
            messages.append({"role": message["role"], "content": message["content"]})

        # Only append the image prompt if images are actually provided 
        if images is not None:
            messages.append({"role": "user", "content": user_message if user_message.strip() else ""})

        return user_message, system_message, messages

    def launch_rag_pipeline(self, base_ip, rag_port, port, engine, model, api_key, temperature, top_p):
        """Launches the RAG pipeline in a new terminal window."""
        print(f"Detected OS: {platform.system()}")
        global RAG_PROCESS
        if RAG_PROCESS is None or RAG_PROCESS.poll() is not None:
            # Create a temporary file to store arguments
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
                args_dict = {
                    "base_ip": base_ip,
                    "rag_port": rag_port,
                    "port": port,
                    "engine": engine,
                    "model": model,
                    "api_key": api_key,
                    "temperature": temperature,
                    "top_p": top_p
                }
                json.dump(args_dict, temp_file)
                temp_file_path = temp_file.name

            # Determine the command to launch the RAG pipeline in a new terminal window
            system = platform.system()
            if system == "Windows":
                command = f'start cmd /c "{sys.executable}" "{os.path.join(os.path.dirname(__file__), "launch_rag.py")}" --args_file="{temp_file_path}"'
            elif system == "Linux":
                # Try different terminal emulators
                terminal_emulators = ["gnome-terminal", "konsole", "xfce4-terminal", "xterm", "uxterm"]
                for emulator in terminal_emulators:
                    if shutil.which(emulator):
                        command = f'{emulator} -e ""{sys.executable}" "{os.path.join(os.path.dirname(__file__), "launch_rag.py")}" --args_file="{temp_file_path}"; read"'
                        break
                else:
                    raise OSError("No supported terminal emulator found.")
            elif system == "Darwin":  # macOS
                command = f'osascript -e \'tell application "Terminal" to do script ""{sys.executable}" "{os.path.join(os.path.dirname(__file__), "launch_rag.py")}" --args_file="{temp_file_path}"; read"'
            else:
                raise OSError(f"Unsupported operating system: {system}")

            # Launch the RAG pipeline in a new terminal window
            RAG_PROCESS = subprocess.Popen(command, shell=True)
            print("RAG pipeline launched in a new terminal window.")
            
            # Clean up the temporary file
            #os.unlink(temp_file_path)    


    def process_chat(self, prompt, engine, model, base_ip, port, document, assistant, rag_port, neg_prompt, embellish_prompt, style_prompt,
                         temperature=0.7, max_tokens=2048, seed=0, random=False, history_steps=10, keep_alive=False, top_k=40, top_p=0.2,
                         repeat_penalty=1.1, stop="", images=None, mode=True, clear_history=True, text_cleanup=True, clear_rag=False, reload_rag=False):
        global RAG_PROCESS
        embellish_content = self.embellish_prompts.get(embellish_prompt, "")
        style_content = self.style_prompts.get(style_prompt, "")
        neg_content = self.neg_prompts.get(neg_prompt, "")

        if images is not None:
            # Normalize tensor 0-255
            img_np = 255.0 * images[0].cpu().numpy()
            # Clip the values to the valid range [0, 255]
            img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))

            # Resize the image if it's too large
            max_size = (1024, 1024)  # Adjust the maximum size as needed
            
            img.thumbnail(max_size)

            # Create a BytesIO object to store the image data
            buffered = io.BytesIO()

            # Save the resized image as PNG
            img.save(buffered, format="PNG")

            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        else:
            base64_image = None

        available_models = self.get_models(engine, base_ip, port)
        if available_models is None or model not in available_models:
            error_message = f"Invalid model selected: {model} for engine {engine}. Available models: {available_models}"
            print(error_message)
            raise ValueError(error_message)

        # Prepare messages
        user_message, system_message, messages = self.prepare_messages(prompt, assistant, images)
        # Get API key
        api_key = self.get_api_key(f"{engine.upper()}_API_KEY", engine)  
        # Upsert the knowledge base if the document has changed
        self.update_knowledge_base(document)
        # Clear chat history if clear_history is True
        if clear_history or history_steps == 0:
            self.chat_history = []
        else:
            self.chat_history = self.chat_history[-history_steps:] if history_steps > 0 else []

        # Update assistant memory only if needed
        if len(self.chat_history) > 0 and self.history_steps > 0 and len(self.chat_history) % self.history_steps == 0:
            self.update_assistant_memory(assistant, messages, engine, base_ip, port, base64_image, model,
                                 system_message, seed, temperature, max_tokens, random,
                                 top_k, top_p, repeat_penalty, stop, keep_alive, api_key)

        # Set stop
        if engine == "ollama":
            if stop == "":
                stop = None
            else:
                stop = ["\n", f"{stop}"]
        elif engine == "lms":
            if stop == "":
                stop = None
            else:
                stop = ["\n", f"{stop}"]
        elif engine == "kobold":
            if stop == "":
                stop = None
            else:
                stop = ["\n\n\n\n\n", f"{stop}"]
        else:
            stop = None
        
        # Clear all contexts
        if clear_rag:
            clear_contexts(assistant)
            self.chat_history = []
            self.document = ""

        # Launch the RAG pipeline if it's not already running
        if RAG_PROCESS is None and mode is False:
            self.launch_rag_pipeline(base_ip, rag_port, port, engine, model, api_key, temperature, top_p)
        
        if reload_rag is True:
            if RAG_PROCESS is not None:
                RAG_PROCESS.kill()
                RAG_PROCESS = None
                self.launch_rag_pipeline(base_ip, rag_port, port, engine, model, api_key, temperature, top_p)

        # Determine which RAG pipeline to use based on keywords in the prompt
        if any(keyword in prompt.lower() for keyword in ["remember", "document", "knowledge", "you", "told", "said", "memory"]) and mode is False:
            try:
                url = f"http://{base_ip}:{self.rag_port}/v1/pw_ai_answer"
                headers = {'accept': '*/*', 'Content-Type': 'application/json'}
                data = {"prompt": prompt, "user": "user"} # Build request data
                print(f"Sending RAG request to: {url}")  # Log request URL
                print(f"Request data: {data}")  # Log request data
                response = requests.post(url, json=data, headers=headers) 

                # Log the response
                print(f"RAG response status code: {response.status_code}")
                print(f"RAG response content: {response.content}") 

                if response.status_code == 200:
                    try:
                        generated_text = response.json()
                        print(f"RAG Response: {generated_text}")
                        self.chat_history.append({"role": "user", "content": user_message})
                        self.chat_history.append({"role": "assistant", "content": generated_text})
                                
                        # Clean up the generated text
                        if text_cleanup:
                            generated_text = process_text(generated_text)
                            generated_text = process_text(generated_text)
                        else:
                            generated_text = generated_text
                            
                        description = generated_text.strip()
                        # Return three values, even in RAG mode
                        return prompt, description, neg_content
                    except json.decoder.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                        print(f"Response content: {response.content}")
                        return prompt, "Error decoding RAG response.", neg_content 
                else:
                    print(f"Error: RAG request failed with status code {response.status_code}")
                    return prompt, "RAG request failed.", neg_content         
                    

            except requests.exceptions.RequestException as e:  # Catch network errors
                print(f"Network error during RAG request: {e}")
                return prompt, "Network error during RAG processing.", neg_content
            except json.decoder.JSONDecodeError as e:  # Catch JSON decoding errors
                print(f"JSON decoding error in RAG response: {e}")
                return prompt, "Error decoding RAG response.", neg_content
            except Exception as e:  # Catch other unexpected exceptions
                print(f"Unexpected exception in RAG processing: {e}")
                return prompt, "Unexpected error during RAG processing.", neg_content

        # If not using RAG features, proceed with regular chat
        else:
            try:
                generated_text, self.chat_history = send_request(engine=engine, base_ip=base_ip, port=port,
                                                                         base64_image=base64_image, model=model,
                                                                         system_message=system_message, user_message=user_message,
                                                                         messages=messages, seed=seed, temperature=temperature,
                                                                         max_tokens=max_tokens, random=random, top_k=top_k,
                                                                         top_p=top_p, repeat_penalty=repeat_penalty,
                                                                         stop=stop, keep_alive=keep_alive,
                                                                         chat_history=self.chat_history, api_key=api_key)
                
                self.chat_history.append({"role": "user", "content": user_message})
                self.chat_history.append({"role": "assistant", "content": generated_text})
                # Clean up the generated text
                if text_cleanup:
                    generated_text = process_text(generated_text)
                else:
                    generated_text = generated_text

                description = generated_text.strip()
               
                
                if mode == False:
                    return prompt, description, neg_prompt
                else:
                    embellish_content = embellish_content.strip()
                    style_content = style_content.strip()
                    combined_prompt = f"{embellish_content} {description} {style_content}".strip()
                    # Return all three outputs
                    return prompt, combined_prompt, neg_content

            except Exception as e:
                print(f"Exception occurred: {e}")
                return "Exception occurred while processing images.", "", ""


NODE_CLASS_MAPPINGS = {"IF_ChatPrompt": IFChatPrompt}
NODE_DISPLAY_NAME_MAPPINGS = {"IF_ChatPrompt": "IF Chat Prompt👨‍💻"}
