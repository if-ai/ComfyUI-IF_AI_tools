import os
import importlib.util
import glob
import shutil
from .IFSaveTextNode import IFSaveText
from .IFDisplayTextNode import IFDisplayText
from .IFChatPromptNode import IFChatPrompt    

NODE_CLASS_MAPPINGS = {
    "IF_SaveText": IFSaveText,
    "IF_DisplayText": IFDisplayText,
    "IF_ChatPrompt": IFChatPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IF_SaveText": "IF Save Text📝",
    "IF_DisplayText": "IF Display Text📟",
    "IF_ChatPrompt": "IF Chat Prompt👨‍💻"
}

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
