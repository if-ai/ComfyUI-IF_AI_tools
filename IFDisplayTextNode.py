class IFDisplayText:
    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {        
                "text": ("STRING", {"forceInput": True}),     
                },
            "hidden": {},
            }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "display_text"
    OUTPUT_NODE = True
    CATEGORY = "ImpactFrames💥🎞️"
    
    def display_text(self, text):
        print("==================")
        print("IF_AI_tool_output:")
        print("==================")
        print(text)
        return {"ui": {"string": [text,]}, "result": (text,)}
    
NODE_CLASS_MAPPINGS = {"IF_DisplayText": IFDisplayText}
NODE_DISPLAY_NAME_MAPPINGS = {"IF_DisplayText": "IF Display Text📟"}


