#Forked from AlekPet/ComfyUI_Custom_Nodes_AlekPet 
class IFDisplayTextNode:
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
    OUTPUT_NODE = True
    FUNCTION = "display_text"
    CATEGORY = "ImpactFrames💥🎞️"

    def display_text(self, text, prompt=None, extra_pnginfo=None):
        return {"ui": {"string": [text,]}, "result": (text,)}
    
NODE_CLASS_MAPPINGS = {"IFDisplayTextNode": IFDisplayTextNode}
NODE_DISPLAY_NAME_MAPPINGS = {"IFDisplayTextNode": "IF Display Text📟"}
