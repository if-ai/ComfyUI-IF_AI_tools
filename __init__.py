import os
import importlib.util
import glob
import shutil
from .IFPromptMkrNode import IFPrompt2Prompt
from .IFImagePromptNode import IFImagePrompt
from .IFSaveTextNode import IFSaveText
from .IFDisplayTextNode import IFDisplayText
from .IFHFDownloadNode import IFHFDownload
from .IFDreamTalkNode import IFDreamTalk
from .IFWhisperSpeechNode import IFWhisperSpeech
from .IFBarkTTSNode import IFBarkTTS
from .IFChatPromptNode import IFChatPrompt    

NODE_CLASS_MAPPINGS = {
    "IF_PromptMkr": IFPrompt2Prompt,
    "IF_ImagePrompt": IFImagePrompt,
    "IF_SaveText": IFSaveText,
    "IF_DisplayText": IFDisplayText,
    "IF_HFDownloadNode": IFHFDownload,
    "IF_DreamTalk": IFDreamTalk,
    "IF_WhisperSpeech": IFWhisperSpeech,
    "IF_BarkTTS": IFBarkTTS,
    "IF_ChatPrompt": IFChatPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IF_PromptMkr": "IF Prompt to Prompt💬",
    "IF_ImagePrompt": "IF Image to Prompt🖼️",
    "IF_SaveText": "IF Save Text📝",
    "IF_DisplayText": "IF Display Text📟",
    "IF_HFDownloadNode": "Hugging Face Download🤗",
    "IF_DreamTalk": "IF DreamTalk🧏🏻",
    "IF_WhisperSpeech": "IF Whisper Speech🌬️",
    "IF_BarkTTS": "IF Bark TTS🐶",
    "IF_ChatPrompt": "IF Chat Prompt👨‍💻"
}

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
