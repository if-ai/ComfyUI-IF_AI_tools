#IFDisplayTextWildcardNode.py
import os
import sys
import yaml
import json
import random
import re
import itertools
import threading
import traceback
from pathlib import Path
import folder_paths
from execution import ExecutionBlocker

class IFDisplayTextWildcard:
    def __init__(self):
        self.wildcards = {}
        self._execution_count = None
        self.wildcard_lock = threading.Lock()
        
        # Initialize paths
        self.base_path = folder_paths.base_path
        self.presets_dir = os.path.join(folder_paths.base_path, "custom_nodes", "ComfyUI-IF_AI_tools", "IF_AI", "presets")
        self.wildcards_dir = os.path.join(self.presets_dir, "wildcards") 

        # Load wildcards
        self.wildcards = self.load_wildcards()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "select": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": sys.maxsize,
                    "step": 1,
                }),
                "counter": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 999999,
                    "step": 1,
                    "display": "number",
                }),
            },
            "optional": {
                "dynamic_prompt": ("STRING", {
                    "multiline": True,
                    "defaultInput": True,
                    "placeholder": "Enter dynamic variables e.g. prefix={val1|val2}"
                }),
                "max_variants": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                }),
                "wildcard_mode": ("BOOLEAN", {
                    "default": False,
                    "display": "button"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("text", "text_list", "count", "selected")
    OUTPUT_IS_LIST = (False, True, False, False)
    FUNCTION = "display_text"
    OUTPUT_NODE = True
    CATEGORY = "ImpactFrames💥🎞️"

    def load_wildcards(self):
        """Load wildcards from YAML/JSON files in the specified directory"""
        wildcard_dict = {}
        wildcards_path = self.wildcards_dir

        def wildcard_normalize(x):
            return x.replace("\\", "/").replace(' ', '-').lower()

        def read_wildcard_file(file_path):
            """Read wildcard definitions from a file"""
            _, ext = os.path.splitext(file_path)
            key = wildcard_normalize(os.path.basename(file_path).split('.')[0])
            try:
                if ext.lower() in ['.yaml', '.yml']:
                    with open(file_path, 'r', encoding="utf-8") as f:
                        yaml_data = yaml.safe_load(f)
                        # Flatten the nested dictionary into wildcard_dict
                        self.flatten_wildcard_dict(yaml_data, key, wildcard_dict)
                elif ext.lower() == '.json':
                    with open(file_path, 'r', encoding="utf-8") as f:
                        json_data = json.load(f)
                        self.flatten_wildcard_dict(json_data, key, wildcard_dict)
                else:
                    print(f"Unsupported file format for wildcards: {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        # Read all files in the wildcards directory
        for file_name in os.listdir(wildcards_path):
            file_path = os.path.join(wildcards_path, file_name)
            if os.path.isfile(file_path):
                read_wildcard_file(file_path)

        print("Loaded Wildcards:")
        for key, values in wildcard_dict.items():
            print(f"{key}: {values}")
        return wildcard_dict

    def flatten_wildcard_dict(self, data, parent_key, wildcard_dict):
        """Flatten nested dictionaries into wildcard_dict with composite keys and aggregate top-level values."""
        def wildcard_normalize(x):
            return x.replace("\\", "/").replace(' ', '-').lower()

        if isinstance(data, dict):
            combined_values = []
            for k, v in data.items():
                new_key = f"{parent_key}/{k}"
                self.flatten_wildcard_dict(v, new_key, wildcard_dict)
                
                # Collect all values from subcategories
                if isinstance(v, dict) or isinstance(v, list):
                    sub_values = self.get_all_nested_values({new_key: v})
                    combined_values.extend(sub_values)
                else:
                    combined_values.append(v)
                
                # Move assignment outside the for loop
                wildcard_dict[parent_key] = combined_values
        elif isinstance(data, list):
            wildcard_dict[parent_key] = data
        else:
            key = wildcard_normalize(parent_key)
            wildcard_dict[key] = [data]

    def get_wildcard_values(self, keyword, pattern_modifier, wildcard_dict):
        """Retrieve wildcard values based on the pattern modifier."""
        keys_to_search = [keyword]

        if pattern_modifier == '/**':
            # Include all nested keys
            keys_to_search = [k for k in wildcard_dict.keys() if k.startswith(f"{keyword}/")]
        elif pattern_modifier == '/*':
            # Include immediate child keys
            keys_to_search = [k for k in wildcard_dict.keys() if k.startswith(f"{keyword}/") and '/' not in k[len(keyword)+1:]]
    
        values = []
        for key in keys_to_search:
            vals = wildcard_dict.get(key, [])
            if isinstance(vals, list):
                values.extend(vals)
            else:
                values.append(vals)
        return values

    def replace_wildcard(self, string, wildcard_dict):
        """Replace wildcards in the given string with appropriate values."""
        pattern = r"__(.+?)(/\*{1,2})?__"  # {{ edit: Updated regex to capture wildcard and pattern modifiers }}
        matches = re.findall(pattern, string)

        replacements_found = False

        for match in matches:
            keyword, pattern_modifier = match
            pattern_modifier = pattern_modifier or ''

            keyword_normalized = keyword.lower().replace('\\', '/').replace(' ', '-')
            
            # Handle pattern modifiers
            if pattern_modifier == '/**':
                values = self.get_wildcard_values(keyword_normalized, '/**', wildcard_dict)
            elif pattern_modifier == '/*':
                values = self.get_wildcard_values(keyword_normalized, '/*', wildcard_dict)
            else:
                values = wildcard_dict.get(keyword_normalized, [])

            if not values:
                print(f"Error: Wildcard __{keyword}{pattern_modifier}__ not found.")
                continue

            replacement = random.choice(values)
            string = string.replace(f"__{keyword}{pattern_modifier}__", replacement, 1)
            replacements_found = True

        return string, replacements_found

    def process(self, text, dynamic_vars, seed=None):
        """Process the text, replacing options and wildcards"""

        if seed is not None:
            random.seed(seed)
        random_gen = random.Random(seed)

        local_wildcard_dict = self.wildcards.copy()
        dynamic_vars_lower = {k.lower(): v for k, v in dynamic_vars.items()}
        local_wildcard_dict.update(dynamic_vars_lower)

        def is_numeric_string(input_str):
            return re.match(r'^-?\d+(\.\d+)?$', input_str) is not None

        def safe_float(x):
            if is_numeric_string(x):
                return float(x)
            else:
                return 1.0

        def replace_options(string):
            replacements_found = False

            def replace_option(match):
                nonlocal replacements_found
                content = match.group(1)
                options = []
                weight_pattern = r'(?:(\d+(?:\.\d+)?)::)?(.*)'
                for opt in content.split('|'):
                    opt = opt.strip()
                    m = re.match(weight_pattern, opt)
                    weight = float(m.group(1)) if m.group(1) else 1.0
                    value = m.group(2).strip()
                    options.append((value, weight))

                # Handle combination syntax
                num_select = 1
                select_sep = ' '
                multi_select_pattern = content.split('$$')
                if len(multi_select_pattern) > 1:
                    range_str = multi_select_pattern[0]
                    options_str = '$$'.join(multi_select_pattern[1:])
                    options = []
                    for opt in options_str.split('|'):
                        opt = opt.strip()
                        m = re.match(weight_pattern, opt)
                        weight = float(m.group(1)) if m.group(1) else 1.0
                        value = m.group(2).strip()
                        options.append((value, weight))

                    if '-' in range_str:
                        min_select, max_select = map(int, range_str.split('-'))
                        num_select = random_gen.randint(min_select, max_select)
                    else:
                        num_select = int(range_str)

                total_weight = sum(weight for value, weight in options)
                normalized_weights = [weight / total_weight for value, weight in options]

                if num_select > len(options):
                    selected_items = [value for value, weight in options]
                    for _ in range(num_select - len(options)):
                        selected_items.append(random_gen.choice(selected_items))
                else:
                    selected_items = random_gen.choices(
                        [value for value, weight in options],
                        weights=normalized_weights,
                        k=num_select
                    )

                replacement = select_sep.join(selected_items)
                replacements_found = True
                return replacement

            pattern = r'\{([^{}]*?)\}'
            replaced_string = re.sub(pattern, replace_option, string)
            return replaced_string, replacements_found

        # Pass 1: replace options
        pass1, is_replaced1 = replace_options(text)

        while is_replaced1:
            pass1, is_replaced1 = replace_options(pass1)

        # Pass 2: replace wildcards using local_wildcard_dict
        text, is_replaced2 = self.replace_wildcard(pass1, local_wildcard_dict)

        stop_unwrap = not is_replaced1 and not is_replaced2

        return text

    def process_text(self, text, dynamic_vars, max_variants, seed=None):
        """Process text replacing wildcards and dynamic variables"""
        output_prompts = []
        base_prompts = [p.strip() for p in text.split("\n") if p.strip()]
        if not base_prompts:
            base_prompts = [""]

        for base_prompt in base_prompts:
            try:
                for _ in range(max_variants):
                    processed_prompt = self.process(base_prompt, dynamic_vars, seed)
                    output_prompts.append(processed_prompt)
            except ValueError as e:
                print(f"Error: {e}")
                continue

        # Ensure unique prompts and respect max_variants
        output_prompts = list(dict.fromkeys(output_prompts))[:max_variants]
        return output_prompts

    def parse_dynamic_variables(self, text):
        """Parse dynamic variables in formats:
        prefix={val1|val2}, **prefix**={val1|val2}, __prefix__={val1|val2}
        """
        variables = {}
        # Match both formats
        patterns = [
            r'(\w+)=\{([^}]+)\}',         # prefix={val1|val2}
            r'\*\*(\w+)\*\*=\{([^}]+)\}', # **prefix**={val1|val2}
            r'__(\w+)__=\{([^}]+)\}'      # __prefix__={val1|val2}
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                category = match.group(1).strip().lower()
                values = [v.strip() for v in match.group(2).split('|')]
                variables[category] = values
        return variables

    def display_text(self, text, select=0, counter=-1, dynamic_prompt="", max_variants=10, wildcard_mode=False):
        """Main node processing function"""
        try:
            # Handle counter
            if self._execution_count is None or self._execution_count > counter:
                self._execution_count = counter
                        
            if self._execution_count == 0:
                return {"ui": {"string": ["Execution blocked: Counter reached 0"]},
                        "result": ExecutionBlocker("Counter reached 0")}

            # Parse dynamic variables if provided
            dynamic_vars = {}
            if dynamic_prompt:
                print(f"Processing dynamic prompt: {dynamic_prompt}")
                dynamic_vars = self.parse_dynamic_variables(dynamic_prompt)
                print(f"Parsed dynamic variables: {dynamic_vars}")

            # Process text
            output_prompts = []
            if wildcard_mode:
                output_prompts = self.process_text(text, dynamic_vars, max_variants)
            else:
                output_prompts = [text]

            # Ensure at least one prompt
            if not output_prompts:
                output_prompts = [text]

            count = len(output_prompts)
            selected = output_prompts[select % count] if count > 0 else text

            # Debug output
            print("\nIF_AI_tool_output:")
            print("==================")
            print(f"Mode: {'Wildcard' if wildcard_mode else 'Normal'}")
            print(f"Counter: {self._execution_count}")
            print(f"Dynamic vars: {dynamic_vars}")
            print(f"Variants generated: {count}")
            for i, p in enumerate(output_prompts):
                print(f"[{i+1}/{count}] {p}")
                print("------------------")
            print("==================")

            # Update counter if needed
            if self._execution_count > 0:
                self._execution_count -= 1

            return {
                "ui": {"string": output_prompts},
                "result": (text, output_prompts, count, selected)
            }

        except Exception as e:
            print(f"Error in display_text: {str(e)}")
            traceback.print_exc()
            return {"ui": {"string": [f"Error: {str(e)}"]},
                    "result": ExecutionBlocker(f"Error: {str(e)}")}
        
    @classmethod
    def IS_CHANGED(cls, text, select, counter, **kwargs):
        return counter

    def get_all_nested_values(self, data):
        """Recursively get all values from nested structure"""
        values = []
        if isinstance(data, dict):
            for v in data.values():
                values.extend(self.get_all_nested_values(v))
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) or isinstance(item, list):
                    values.extend(self.get_all_nested_values(item))
                else:
                    values.append(item)
        else:
            values.append(data)
        return values

    def get_root_values(self, data):
        """Get only root level values"""
        values = []
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    values.extend(v)
                elif isinstance(v, str):
                    values.append(v)
        elif isinstance(data, list):
            values.extend(data)
        elif isinstance(data, str):
            values.append(data)
        return values

