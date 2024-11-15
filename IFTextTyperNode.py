class IFTextTyper:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "output_text"
    OUTPUT_NODE = True
    CATEGORY = "ImpactFrames💥🎞️"

    def output_text(self, text):
        return (text,)

NODE_CLASS_MAPPINGS = {
    "IF_TextTyper": IFTextTyper
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IF_TextTyper": "IF Text Typer✍️"
}