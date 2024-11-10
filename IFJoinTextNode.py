class IFJoinText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "separator": ("STRING", {
                    "multiline": False,
                    "default": " ",
                    "placeholder": "Text to insert between joined strings"
                }),
            },
            "optional": {
                "text1": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "forceInput": True,
                    "placeholder": "First text input"
                }),
                "text2": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "forceInput": True,
                    "placeholder": "Second text input"
                }),
                "text3": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "forceInput": True,
                    "placeholder": "Third text input"
                }),
                "text4": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "forceInput": True,
                    "placeholder": "Fourth text input"
                }),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "join_text"
    CATEGORY = "ImpactFrames💥🎞️"
    
    def join_text(self, separator=" ", text1="", text2="", text3="", text4=""):
        # Collect all non-empty text inputs
        texts = [t for t in [text1, text2, text3, text4] if t.strip()]
        
        # Join texts with separator
        result = separator.join(texts)
        
        # Print for debugging
        print("==================")
        print("IF_JoinText output:")
        print("==================")
        print(result)
        
        return (result,)

NODE_CLASS_MAPPINGS = {
    "IF_JoinText": IFJoinText
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IF_JoinText": "IF Join Text 📝"
}