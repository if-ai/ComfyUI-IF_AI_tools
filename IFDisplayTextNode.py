import sys
import logging
from typing import Optional

# Initialize logger
logger = logging.getLogger(__name__)

class IFDisplayText:
    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {        
                "text": ("STRING", {"forceInput": True}),
                "select": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": sys.maxsize,  # No practical upper limit
                    "step": 1,
                    "tooltip": "Select which line to output (cycles through available lines)"
                }),     
            },
            "hidden": {},
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("text", "text_list", "count", "selected")
    OUTPUT_IS_LIST = (False, True, False, False)
    FUNCTION = "display_text"
    OUTPUT_NODE = True
    CATEGORY = "ImpactFrames💥🎞️"
    
    def display_text(self, text: Optional[str], select):
        if text is None:
            logger.error("Received None for text input in display_text.")
            return ""  # Or handle appropriately

        print("==================")
        print("IF_AI_tool_output:")
        print("==================")
        print(text)
        
        # Split text into lines and filter out empty lines
        text_list = [line.strip() for line in text.split('\n') if line.strip()]
        count = len(text_list)
        
        # Select line using modulo to handle cycling
        if count == 0:
            selected = text  # If no valid lines, return original text
        else:
            selected = text_list[select % count]
        
        # Return both UI update and the multiple outputs
        return {
            "ui": {"string": [text]}, 
            "result": (
                text,        # complete text
                text_list,   # list of individual lines as separate string outputs
                count,       # number of lines
                selected    # selected line based on select input
            )
        }

NODE_CLASS_MAPPINGS = {"IF_DisplayText": IFDisplayText}
NODE_DISPLAY_NAME_MAPPINGS = {"IF_DisplayText": "IF Display Text📟"}


