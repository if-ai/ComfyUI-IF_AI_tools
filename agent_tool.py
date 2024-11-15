from typing import Dict, Any
import importlib.util
import folder_paths
import os
import sys

class AgentTool:
    def __init__(self, name, description, system_prompt, default_engine, default_model, 
                 default_temperature, default_max_tokens, python_class, python_function=None, output_type=None):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.default_engine = default_engine
        self.default_model = default_model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.python_class = python_class
        self.python_function = python_function
        self.output_type = output_type
        self._class_instance = None
        self._function = None
        self.comfy_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def load(self):
        # Construct the path to the ComfyUI-IF_AI_tools directory
        if_ai_tools_dir = os.path.join(folder_paths.base_path, "custom_nodes",  "ComfyUI-IF_AI_tools")
        
        # Add the ComfyUI-IF_AI_tools directory to sys.path
        if if_ai_tools_dir not in sys.path:
            sys.path.insert(0, if_ai_tools_dir)

        # Import the module
        module_name = self.python_class.split('.')[0]
        file_path = os.path.join(if_ai_tools_dir, f"{module_name}.py")
        #print(f"Attempting to load module from: {file_path}")
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            #print(f"Successfully loaded module from: {file_path}")
        except Exception as e:
            print(f"Error loading module {module_name}: {str(e)}")
            print(f"sys.path: {sys.path}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"File exists: {os.path.exists(file_path)}")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    print(f"File contents:\n{f.read()}")
            return

        # Get the class and create an instance
        class_name = self.python_class.split('.')[-1]
        try:
            class_ = getattr(module, class_name)
            self._class_instance = class_(self.name, self.description, self.system_prompt)
            #print(f"Successfully created instance of {class_name}")
        except AttributeError as e:
            print(f"Warning: Could not find class {class_name} in module {module_name}. Error: {e}")
            return

        # Get the function if specified
        if self.python_function:
            try:
                self._function = getattr(self._class_instance, self.python_function)
                #print(f"Successfully loaded function {self.python_function}")
            except AttributeError as e:
                print(f"Warning: Could not find function {self.python_function} in class {class_name}. Error: {e}")
                return

    def execute(self, args):
        if self._function:
            return self._function(args)
        elif self._class_instance and hasattr(self._class_instance, 'execute'):
            return self._class_instance.execute(args)
        else:
            raise NotImplementedError("No execution method available for this tool")
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "args": {
                        "type": "string",
                        "description": "Arguments for the function"
                    }
                },
                "required": ["args"]
            }
        }