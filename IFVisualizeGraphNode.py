import json
import networkx as nx
import os

class IFVisualizeGraphNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "graph_data": ("STRING", {"tooltip": "GraphML file path"}),
            },
            "optional": {
                "layout": (["spring", "circular", "random", "shell", "spectral"], {"default": "spring"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "visualize_graph"
    CATEGORY = "ImpactFrames💥🎞️"
    OUTPUT_NODE = True

    def visualize_graph(self, graph_data, layout="spring"):
        print(f"Visualizing graph: {graph_data}, layout: {layout}")
        try:
            if not os.path.exists(graph_data):
                print(f"GraphML file not found: {graph_data}")
                return {}, {"ui": {"error": f"GraphML file not found: {graph_data}"}}

            G = nx.read_graphml(graph_data)
            graph_json = json.dumps(nx.node_link_data(G))
            print(f"Graph JSON (first 100 chars): {graph_json[:100]}...")
            
            return {}, {"ui": {"graph": graph_json, "layout": layout}}
        
        except Exception as e:
            import traceback
            error_message = f"Error: {str(e)}\nTraceback:\n{traceback.format_exc()}"
            print(error_message)
            return {}, {"ui": {"error": error_message}}

NODE_CLASS_MAPPINGS = {
    "IF_VisualizeGraph": IFVisualizeGraphNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IF_VisualizeGraph": "IF Visualize Graph🕸️"
} 