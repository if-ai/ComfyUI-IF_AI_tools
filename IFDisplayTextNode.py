class IFDisplayText:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {        
                "text": ("STRING", {"forceInput": True}),     
                },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
            }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "display_text"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    CATEGORY = "ImpactFrames💥🎞️"
    
    def display_text(self, text, prompt=None, extra_pnginfo=None):
        return {"ui": {"string": [text,]}, "result": (text,)}
    
NODE_CLASS_MAPPINGS = {"IF_DisplayText": IFDisplayText}
NODE_DISPLAY_NAME_MAPPINGS = {"IF_DisplayText": "IF Display Text📟"}


