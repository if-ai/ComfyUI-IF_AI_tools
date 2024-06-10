import argparse
import json
from rag_module import run_pathway_pipeline
import threading


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--args_file", required=True, help="Path to the JSON file containing the arguments")
    args = parser.parse_args()

    with open(args.args_file, 'r') as f:
        args_dict = json.load(f)

    threading.Thread(target= run_pathway_pipeline(
        args_dict["base_ip"],
        args_dict["rag_port"],
        args_dict["port"],
        args_dict["engine"],
        args_dict["model"],
        args_dict["api_key"],
        args_dict["temperature"],
        args_dict["top_p"]
    ), daemon=True).start()

    print(f"Pathway RAG server started at http://{args_dict['base_ip']}:{args_dict['rag_port']}")