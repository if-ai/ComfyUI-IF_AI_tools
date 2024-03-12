import os
from .IFPromptMkrNode import IFPromptMkrNode
from .IFImagePromptNode import IFImagePromptNode
from .IFSaveTextNode import IFSaveTextNode
from .IFDisplayTextNode import IFDisplayTextNode

NODE_CLASS_MAPPINGS = {
    "IFPromptMkrNode": IFPromptMkrNode,
    "IFImagePromptNode": IFImagePromptNode,
    "IFSaveTextNode": IFSaveTextNode,
    "IFDisplayTextNode": IFDisplayTextNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IFPromptMkrNode": "IF Prompt to Prompt💬",
    "IFImagePromptNode": "IF Image to Prompt🖼️",
    "IFSaveTextNode": "IF Save Text📝",
    "IFDisplayTextNode": "IF Display Text📟"
}

EXTENSION_WEB_DIRS = {
    "ComfyUI-IF_AI_tools": os.path.join(os.path.dirname(__file__), "js")
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "EXTENSION_WEB_DIRS"]
