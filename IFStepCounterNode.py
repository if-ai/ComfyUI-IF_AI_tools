class IFCounter:
    def __init__(self):
        self.counters = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "number_type": (["integer", "float"],),
                "mode": (["increment", "decrement", "increment_to_stop", "decrement_to_stop"],),
                "start": ("FLOAT", {"default": 0, "min": -99999999999999, "max": 99999999999999, "step": 0.01}),
                "stop": ("FLOAT", {"default": 0, "min": -99999999999999, "max": 99999999999999, "step": 0.01}),
                "step": ("FLOAT", {"default": 1, "min": 0, "max": 99999, "step": 0.01}),
            },
            "optional": {
                "reset_bool": ("NUMBER",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    RETURN_TYPES = ("NUMBER", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("number", "float", "int", "string")
    FUNCTION = "increment_number"
    
    CATEGORY = "ImpactFrames💥🎞️"
    
    def increment_number(self, number_type, mode, start, stop, step, unique_id, reset_bool=0):
        # Initialize counter
        counter = int(start) if mode == 'integer' else start
        if self.counters.__contains__(unique_id):
            counter = self.counters[unique_id]
        
        # Handle reset
        if round(reset_bool) >= 1:
            counter = start
        
        # Process counter based on mode
        if mode == 'increment':
            counter += step
        elif mode == 'decrement':  # Fixed typo in 'deccrement'
            counter -= step
        elif mode == 'increment_to_stop':
            counter = counter + step if counter < stop else counter
        elif mode == 'decrement_to_stop':
            counter = counter - step if counter > stop else counter
        
        # Store counter
        self.counters[unique_id] = counter
        
        # Prepare results
        result = int(counter) if number_type == 'integer' else float(counter)
        
        # Convert to string with appropriate formatting
        if number_type == 'integer':
            string_result = str(int(counter))
        else:
            # Remove trailing zeros and decimal point if it's a whole number
            string_result = f"{float(counter):g}"
        
        return (result, float(counter), int(counter), string_result)

NODE_CLASS_MAPPINGS = {"IF_StepCounter": IFCounter}
NODE_DISPLAY_NAME_MAPPINGS = {"IF_StepCounter": "IF Step Counter 🔢"}