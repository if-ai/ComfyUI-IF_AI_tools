import os
import importlib.util
import glob
import shutil
import sys
import folder_paths
from aiohttp import web

from .IFChatPromptNode import IFChatPrompt
from .IFImagePromptNode import IFImagePrompt
from .IFPromptMkrNode import IFPrompt2Prompt
from .IFDisplayTextWildcardNode import IFDisplayTextWildcard
from .IFSaveTextNode import IFSaveText
from .IFDisplayTextNode import IFDisplayText
from .IFDisplayOmniNode import IFDisplayOmni
from .IFTextTyperNode import IFTextTyper
from .IFVisualizeGraphNode import IFVisualizeGraphNode
from .IFStepCounterNode import IFCounter
from .IFJoinTextNode import IFJoinText
from .IFLoadImagesNodeS import IFLoadImagess
from .send_request import *

# Try to import omost from the current directory
# Add the current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from .omost import omost_function
    print("Successfully imported omost_function from omost.py in the current directory")
except ImportError as e:
    print(f"Error importing omost from current directory: {e}")
    
    # If import fails, try to import from the parent directory
    parent_dir = os.path.dirname(current_dir)
    parent_dir_name = os.path.basename(parent_dir)
    if parent_dir_name == 'ComfyUI-IF_AI_tools':
        sys.path.insert(0, parent_dir)
        try:
            from omost import omost_function
            print(f"Successfully imported omost_function from {parent_dir}/omost.py")
        except ImportError as e:
            print(f"Error importing omost from parent directory: {e}")
            print(f"Current sys.path: {sys.path}")
            raise
class OmniType(str):
    """A special string type that acts as a wildcard for universal input/output. 
       It always evaluates as equal in comparisons."""
    def __ne__(self, __value: object) -> bool:
        return False
    
OMNI = OmniType("*")
                       
NODE_CLASS_MAPPINGS = {
    "IF_ChatPrompt": IFChatPrompt, 
    "IF_PromptMkr": IFPrompt2Prompt,
    "IF_ImagePrompt": IFImagePrompt,
    "IF_SaveText": IFSaveText,
    "IF_DisplayText": IFDisplayText,
    "IF_DisplayTextWildcard": IFDisplayTextWildcard,
    "IF_DisplayOmni": IFDisplayOmni,
    "IF_TextTyper": IFTextTyper,
    "IF_VisualizeGraph": IFVisualizeGraphNode,
    "IF_StepCounter": IFCounter,
    "IF_JoinText": IFJoinText,
    "IF_LoadImagesS": IFLoadImagess,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IF_ChatPrompt": "IF Chat Prompt👨‍💻",
    "IF_PromptMkr": "IF Prompt Maker🎨",
    "IF_ImagePrompt": "IF Image to Prompt🖼️",
    "IF_SaveText": "IF Save Text📝",
    "IF_DisplayText": "IF Display Text📟",
    "IF_DisplayTextWildcard": "IF Display Text Wildcard📟",
    "IF_DisplayOmni": "IF Display Omni🔍",
    "IF_TextTyper": "IF Text Typer✍️",
    "IF_VisualizeGraph": "IF Visualize Graph🕸️",
    "IF_StepCounter": "IF Step Counter 🔢",
    "IF_JoinText": "IF Join Text 📝",
    "IF_LoadImagesS": "IF Load Images S 🖼️"
}

WEB_DIRECTORY = "./web"
__all__ = [
    "NODE_CLASS_MAPPINGS", 
    "NODE_DISPLAY_NAME_MAPPINGS", 
    "WEB_DIRECTORY", 
    "omost_function"
    ]
