import json

class AlwaysEqualProxy:
    def __init__(self, value):
        self.value = value

    def __eq__(self, text):
        return True

class IFDisplayOmni:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {"omni_input": ("OMNI", {})},
            "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("OMOST_CANVAS_CONDITIONING", "STRING")
    RETURN_NAMES = ("canvas_conditioning", "text_output")
    INPUT_IS_LIST = True
    OUTPUT_NODE = True
    FUNCTION = "display_omni"
    CATEGORY = "ImpactFrames💥🎞️"

    def display_omni(self, unique_id=None, extra_pnginfo=None, **kwargs):
        values = []
        canvas_conditioning = None
        text_output = ""

        if "omni_input" in kwargs:
            for val in kwargs['omni_input']:
                try:
                    if isinstance(val, str):
                        values.append(val)
                        text_output = val
                    elif isinstance(val, list) and all(isinstance(item, dict) for item in val):
                        # This is likely the canvas conditioning
                        canvas_conditioning = val
                        values.append(json.dumps(val))
                    else:
                        json_val = json.dumps(val)
                        values.append(str(json_val))
                        text_output = str(json_val)
                except Exception:
                    values.append(str(val))
                    text_output = str(val)

        if unique_id is not None and extra_pnginfo is not None:
            if isinstance(extra_pnginfo, list) and len(extra_pnginfo) > 0:
                extra_pnginfo = extra_pnginfo[0]
            
            if isinstance(extra_pnginfo, dict) and "workflow" in extra_pnginfo:
                workflow = extra_pnginfo["workflow"]
                node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id), None)
                if node:
                    node["widgets_values"] = [values]
            else:
                print("Error: extra_pnginfo is not in the expected format")
        else:
            print("Error: unique_id or extra_pnginfo is None")

        return {
            "ui": {"text": values},
            "result": (canvas_conditioning, text_output)
        }