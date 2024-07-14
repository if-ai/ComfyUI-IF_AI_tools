import os
import importlib.util
import glob
import shutil
from .IFPromptMkrNode import IFPrompt2Prompt
from .IFImagePromptNode import IFImagePrompt
from .IFSaveTextNode import IFSaveText
from .IFDisplayTextNode import IFDisplayText
from .IFChatPromptNode import IFChatPrompt
from .IFDisplayOmniNode import IFDisplayOmni

class OmniType(str):
    """A special string type that acts as a wildcard for universal input/output. 
       It always evaluates as equal in comparisons."""
    def __ne__(self, __value: object) -> bool:
        return False
    
OMNI = OmniType("*")


NODE_CLASS_MAPPINGS = {
    "IF_PromptMkr": IFPrompt2Prompt,
    "IF_ImagePrompt": IFImagePrompt,
    "IF_SaveText": IFSaveText,
    "IF_DisplayText": IFDisplayText,
    "IF_ChatPrompt": IFChatPrompt,
    "IF_DisplayOmni": IFDisplayOmni,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IF_PromptMkr": "IF Prompt to Prompt💬",
    "IF_ImagePrompt": "IF Image to Prompt🖼️",
    "IF_SaveText": "IF Save Text📝",
    "IF_DisplayText": "IF Display Text📟",
    "IF_ChatPrompt": "IF Chat Prompt👨‍💻",
    "IF_DisplayOmni": "IF Display Omni🔍",
    #"IF_AI_Agent": "IF AI Agent 🤖"
}

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
