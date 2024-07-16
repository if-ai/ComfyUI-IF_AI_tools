import json
import os
import io
import torch
import base64
import importlib
import numpy as np
from PIL import Image
from server import PromptServer

from typing import Dict, Any
from .get_api import get_api_key
from .text_cleanup import process_text
from .send_request import send_request
from .agent_tool import AgentTool

from aiohttp import web
from .get_models import get_models


@PromptServer.instance.routes.post("/IF_ChatPrompt/get_models")
async def get_models_endpoint(request):
    data = await request.json()
    engine = data.get("engine")
    base_ip = data.get("base_ip")
    port = data.get("port")
    
    models = get_models(engine, base_ip, port)
    return web.json_response(models)


class IFChatPrompt:
    RETURN_TYPES = ("STRING", "STRING", "STRING", "OMNI")
    RETURN_NAMES = ("Question", "Response", "Negative", "Tool_Output")
    FUNCTION = "process_chat"
    OUTPUT_NODE = True
    CATEGORY = "ImpactFrames💥🎞️"

    def __init__(self):        
        self.base_ip = "localhost"
        self.port = "11434"
        self.engine = "ollama"
        self.model = ""
        self.assistant = "Cortana"
        self.comfy_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.presets_dir = os.path.join(self.comfy_dir, "input", "IF_AI", "presets")
        self.stop_file = os.path.join(self.presets_dir, "stop_strings.json")
        self.assistants_file = os.path.join(self.presets_dir, "assistants.json")
        self.neg_prompts_file = os.path.join(self.presets_dir, "neg_prompts.json")
        self.embellish_prompts_file = os.path.join(self.presets_dir, "embellishments.json")
        self.style_prompts_file = os.path.join(self.presets_dir, "style_prompts.json")
        self.agents_dir = os.path.join(self.presets_dir, "agents")
        self.agent_tools = self.load_agent_tools()
        self.stop_strings = self.load_presets(self.stop_file)
        self.assistants = self.load_presets(self.assistants_file)
        self.neg_prompts = self.load_presets(self.neg_prompts_file)
        self.embellish_prompts = self.load_presets(self.embellish_prompts_file)
        self.style_prompts = self.load_presets(self.style_prompts_file)
        self.keep_alive = False
        self.seed = 94687328150
        self.messages = []
        self.history_steps = 10
        self.external_api_key = ""
        self.tool_input = ""
        self.prime_directives = None
        #self.agent_tools = self.load_agent_tools()
        #self.agent_tools["omost_tool"] = OmostTool()

    def load_presets(self, file_path):
        with open(file_path, 'r') as f:
            presets = json.load(f)
        return presets
    
    def load_agent_tools(self):
        agent_tools = {}
        #print(f"Agents directory: {self.agents_dir}")
        #print(f"Files in agents directory: {os.listdir(self.agents_dir)}")
        for filename in os.listdir(self.agents_dir):
            if filename.endswith('.json'):
                full_path = os.path.join(self.agents_dir, filename)
                #print(f"Loading agent tool from: {full_path}")
                with open(full_path, 'r') as f:
                    try:
                        data = json.load(f)
                        #print(f"Loaded data: {data}")
                        # Add a default value for output_type if it's not in the JSON
                        if 'output_type' not in data:
                            data['output_type'] = None
                        agent_tool = AgentTool(**data)
                        agent_tool.load()  # This will create the class instance
                        if agent_tool._class_instance is not None:  # Check if the instance was created successfully
                            if agent_tool.python_function:
                                agent_tools[agent_tool.name] = agent_tool
                                #print(f"Successfully loaded agent tool: {agent_tool.name}")
                            else:
                                print(f"Warning: Agent tool {agent_tool.name} in {filename} does not have a python_function defined.")
                        else:
                            print(f"Failed to create class instance for {filename}")
                    except json.JSONDecodeError:
                        print(f"Error: Invalid JSON in {filename}")
                    except Exception as e:
                        print(f"Error loading {filename}: {str(e)}")
        #print(f"Loaded agent tools: {list(agent_tools.keys())}")
        return agent_tools


    @classmethod
    def INPUT_TYPES(cls):
        node = cls()
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "base_ip": ("STRING", {"default": node.base_ip}),
                "port": ("STRING", {"default": node.port}),
                "engine": (["llamacpp", "ollama", "kobold", "lmstudio", "textgen", "groq", "gemini", "openai", "anthropic", "mistral"], {"default": node.engine}),
                #"model": (node.get_models("node.engine", node.base_ip, node.port), {}), 
                "model": ((), {}),
                #"model": ("STRING", {"default": ""}),
                "assistant": ([name for name in node.assistants.keys()], {"default": node.assistant}),
            },
            "optional": {
                "images": ("IMAGE", ),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 2048, "min": 0, "max": 0xffffffffffffffff}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100}),
                "top_p": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.1}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 10.0, "step": 0.1}),
                "stop_string": ([name for name in node.stop_strings.keys()], {}),
                "seed": ("INT", {"default": 94687328150, "min": 0, "max": 0xffffffffffffffff}),
                "random": ("BOOLEAN", {"default": False, "label_on": "Seed", "label_off": "Temperature"}),
                "embellish_prompt": ([name for name in node.embellish_prompts.keys()], {}),
                "style_prompt": ([name for name in node.style_prompts.keys()], {}),
                "neg_prompt": ([name for name in node.neg_prompts.keys()], {}),
                "clear_history": ("BOOLEAN", {"default": True, "label_on": "Clear History", "label_off": "Keep History"}),
                "history_steps": ("INT", {"default": 10, "min": 0, "max": 0xffffffffffffffff}),
                "keep_alive": ("BOOLEAN", {"default": False, "label_on": "Keeps_Model", "label_off": "Unloads_Model"}),
                "text_cleanup": ("BOOLEAN", {"default": True, "label_on": "Apply", "label_off": "Raw Text"}),
                "mode": ("BOOLEAN", {"default": True, "label_on": "Mode: SD", "label_off": "Mode: Chat"}),
                "external_api_key": ("STRING", {"default": "", "multiline": False}), 
                "tool": (["None"] + [name for name in node.agent_tools.keys()], {"default": "None"}),
                "tool_input": ("OMNI", {"default": None}),
                "prime_directives": ("STRING", {"forceInput": True}),
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
    
    def get_models(self, engine, base_ip, port):
        from .get_models import get_models as fetch_models
        return fetch_models(engine, base_ip, port)

    def process_chat(self, prompt, engine, model, base_ip, port, assistant, neg_prompt, embellish_prompt, style_prompt, external_api_key,
                 temperature=0.7, max_tokens=2048, seed=0, random=False, history_steps=10, keep_alive=False, top_k=40, top_p=0.2,
                 repeat_penalty=1.1, stop_string=None, images=None, mode=True, clear_history=False, text_cleanup=True,
                 tool=None, tool_input=None, prime_directives=None):
        
        if prime_directives != None:
            system_message_str = prime_directives
        else:  
            system_message = self.assistants.get(assistant, "")
            system_message_str = json.dumps(system_message)
        #print("system_message", system_message_str)
        

        # Handle history
        if clear_history:
            self.messages = []
        elif history_steps > 0:
            self.messages = self.messages[-history_steps:]

        # Process image if provided
        base64_image = None
        if images is not None:
            img_np = 255.0 * images[0].cpu().numpy()
            img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
            max_size = (1024, 1024)
            img.thumbnail(max_size)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Handle API key
        if not external_api_key:
            api_key = get_api_key(f"{engine.upper()}_API_KEY", engine)
        else:
            api_key = external_api_key

        # Validate model
        available_models = get_models(engine, base_ip, port)
        if available_models is None or model not in available_models:
            error_message = f"Invalid model selected: {model} for engine {engine}. Available models: {available_models}"
            print(error_message)
            raise ValueError(error_message)

        # Handle stop sequences
        if stop_string is None or stop_string == "None":
            stop_content = None
        else:
            stop_content = self.stop_strings.get(stop_string, None)
        
        # For Ollama, we need to pass the stop content as is
        stop = stop_content

        # For other engines, we might need to adjust the stop content
        if engine not in ["ollama", "llamacpp", "vllm", "lmstudio", "gemeni"]:
            if engine == "kobold":
                stop = stop_content + ["\n\n\n\n\n"] if stop_content else ["\n\n\n\n\n"]
            elif engine == "mistral":
                stop = stop_content + ["\n\n"] if stop_content else ["\n\n"]
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
                # Prepare the tool data for the API call
                #tool_data = selected_tool.to_dict()
                #print (f"this is the content of tool data dict {tool_data}")
                # Send request to LLM for tool execution
                generated_text = send_request(
                    engine=engine,
                    base_ip=base_ip,
                    port=port,
                    base64_image=base64_image,
                    model=model,
                    system_message=system_prompt,
                    user_message=tool_message,
                    messages=self.messages,
                    seed=seed,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    random=random,
                    top_k=top_k,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    stop=stop,
                    keep_alive=keep_alive,
                    api_key=api_key
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
                self.messages.append({"role": "user", "content": prompt})
                self.messages.append({
                    "role": "assistant",
                    "content": json.dumps(tool_result) if isinstance(tool_result, dict) else str(tool_result)
                })

                # Print messages to console
                #for message in self.messages:
                #    print(f"{message['role']}: {message['content']}")

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

                if mode:
                    embellish_content = self.embellish_prompts.get(embellish_prompt, "").strip()
                    style_content = self.style_prompts.get(style_prompt, "").strip()
                    combined_prompt = f"{embellish_content} {generated_text} {style_content}".strip()
                    return prompt, combined_prompt, self.neg_prompts.get(neg_prompt, ""), tool_output
                else:
                    return prompt, generated_text, "", tool_output
            else:
                # Process the request for normal chat
                generated_text = send_request(
                    engine=engine,
                    base_ip=base_ip,
                    port=port,
                    base64_image=base64_image,
                    model=model,
                    system_message=system_message_str,
                    user_message=prompt,
                    messages=self.messages,
                    seed=seed,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    random=random,
                    top_k=top_k,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    stop=stop,
                    keep_alive=keep_alive,
                    api_key=api_key
                )
                
                tool_output = None

                # Update messages
                self.messages.append({"role": "user", "content": prompt})
                self.messages.append({"role": "assistant", "content": generated_text})

                #print messages to console
                #for message in self.messages:
                #    print(f"{message['role']}: {message['content']}")

                # Process the generated text
                if text_cleanup:
                    generated_text = process_text(generated_text)

                description = generated_text.strip()
                print (f"Generated text: {description}")

                if mode:
                    embellish_content = self.embellish_prompts.get(embellish_prompt, "").strip()
                    style_content = self.style_prompts.get(style_prompt, "").strip()
                    combined_prompt = f"{embellish_content} {description} {style_content}".strip()
                    return prompt, combined_prompt, self.neg_prompts.get(neg_prompt, ""), tool_output
                else:
                    return prompt, description, self.neg_prompts.get(neg_prompt, ""), tool_output

        except Exception as e:
            print(f"Exception occurred: {e}")
            return "Exception occurred while processing request.", "", "", None

NODE_CLASS_MAPPINGS = {"IF_ChatPrompt": IFChatPrompt}
NODE_DISPLAY_NAME_MAPPINGS = {"IF_ChatPrompt": "IF Chat Prompt👨‍💻"}