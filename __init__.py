import os
from .IFPromptMkrNode import IFPrompt2Prompt
from .IFImagePromptNode import IFImagePrompt
from .IFSaveTextNode import IFSaveText
from .IFDisplayTextNode import IFDisplayText

NODE_CLASS_MAPPINGS = {
    "IF_PromptMkr": IFPrompt2Prompt,
    "IF_ImagePrompt": IFImagePrompt,
    "IF_SaveText": IFSaveText,
    "IF_DisplayText": IFDisplayText
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IF_PromptMkr": "IF Prompt to Prompt💬",
    "IF_ImagePrompt": "IF Image to Prompt🖼️",
    "IF_SaveText": "IF Save Text📝",
    "IF_DisplayText": "IF Display Text📟"
}

EXTENSION_WEB_DIRS = {
    "ComfyUI-IF_AI_tools": os.path.join(os.path.dirname(__file__), "js")
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "EXTENSION_WEB_DIRS"]
