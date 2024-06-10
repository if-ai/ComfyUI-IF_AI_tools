from contextlib import redirect_stdout
from datetime import date
import os
import json
import tempfile
import shutil
import textwrap
import pathway as pw
import pandas as pd
import logging
from aiohttp import web
import argparse
import importlib.util
import time
import asyncio
import websockets
import threading

from pathway.xpacks.llm import embedders, parsers, splitters, prompts, llms
from pathway.xpacks.llm.question_answering import AdaptiveRAGQuestionAnswerer, BaseRAGQuestionAnswerer
from pathway.udfs import ExponentialBackoffRetryStrategy, DiskCache, DefaultCache, FixedDelayRetryStrategy
from pathway.xpacks.llm.llms import prompt_chat_single_qa
from pathway.xpacks.llm.vector_store import VectorStoreServer
from pathway.stdlib.indexing import default_vector_document_index
from pathway.xpacks.llm.question_answering import (
    answer_with_geometric_rag_strategy_from_index,
)

comfy_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KNOWLEDGE_BASE_DIR = os.path.join(comfy_dir, "output", "IF_AI", "knowledge_base")
ASSISTANT_MEMORY_DIR = os.path.join(comfy_dir, "output", "IF_AI", "assistant_data")
CHAT_HISTORY_DIR = os.path.join(comfy_dir, "output", "IF_AI", "chat_history")
max_tokens: int = 60
device: str = "cpu"
embedding_model = "avsolatorio/GIST-small-Embedding-v0"

# Global variables to manage the RAG pipeline
vector_database = None
embedder = None
text_splitter = None
parser = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def clear_contexts(assistant):
    """Clears context for the given assistant."""
    global rag_pipeline, vector_database

    if vector_database is not None:
        vector_database.clear_context()

    chat_history_dir = os.path.join(CHAT_HISTORY_DIR, f"{assistant}_chat_history")
    shutil.rmtree(chat_history_dir, ignore_errors=True)
    os.makedirs(chat_history_dir, exist_ok=True)
    open(os.path.join(chat_history_dir, "placeholder.txt"), 'w').close()

    assistant_memory_dir = os.path.join(ASSISTANT_MEMORY_DIR, f"{assistant}_memory")
    shutil.rmtree(assistant_memory_dir, ignore_errors=True)
    os.makedirs(assistant_memory_dir, exist_ok=True)
    open(os.path.join(assistant_memory_dir, "placeholder.txt"), 'w').close()

    print(f"Contexts cleared for assistant: {assistant}")

# Create a global threading.Event object
RAG_READY_EVENT = threading.Event()

# Modify the run_pathway_pipeline function
def run_pathway_pipeline(base_ip, rag_port, port, engine, model, api_key, temperature, top_p):
    """Initializes the RAG pipeline and sets the ready event."""

    documents = []
    documents.append(
        pw.io.fs.read(
            path=KNOWLEDGE_BASE_DIR,
            format="binary",
            with_metadata=True,
            mode="streaming",
        )
    )

    parser = parsers.ParseUnstructured()
    sources = documents

    text_splitter = splitters.TokenCountSplitter(max_tokens=400)
    embedder = embedders.SentenceTransformerEmbedder(
        embedding_model, call_kwargs={"show_progress_bar": False}
    )

    vector_database = VectorStoreServer(
        *sources,
        embedder=embedder,
        splitter=text_splitter,
        parser=parser,
    )

    rag_pipeline = BaseRAGQuestionAnswerer(
        llm=configure_llm(base_ip, port, engine, model, temperature, api_key, top_p),
        indexer=vector_database,
        search_topk=6,
        short_prompt_template=prompts.prompt_qa,
    )

    rag_pipeline.build_server(host=base_ip, port=int(rag_port))
    
    rag_pipeline.run_server(terminate_on_error=False, with_cache=True)
    # Set the event to signal that the RAG pipeline is ready
    RAG_READY_EVENT.set()
    print("RAG pipeline initialized and ready.")
    return rag_pipeline

def configure_llm(base_ip, port, engine, model, temperature, api_key, top_p):
    """Configures the LLM based on the provided parameters."""

    if engine == "kobold" or engine == "textgen" or engine == "lms":
        chat = llms.LiteLLMChat(
            model=f"openai/{model}",
            temperature=temperature,
            top_p=top_p,
            api_base=f"http://{base_ip}:{port}",  # local deployment
            custom_llm_provider="openai",  # Tell LiteLLM to use OpenAI API format
            retry_strategy=ExponentialBackoffRetryStrategy(max_retries=6), 
        )
    elif engine == "ollama":
        chat = llms.LiteLLMChat(
            model=f"ollama/{model}", 
            temperature=temperature,
            top_p=top_p,
            api_base=f"http://{base_ip}:{port}",  # local deployment
            retry_strategy=ExponentialBackoffRetryStrategy(max_retries=6), 
        )
    elif engine == "groq":
        api_key = api_key 
        chat = llms.LiteLLMChat(
            model=f"groq/{model}",
            temperature=temperature,
            top_p=top_p,
            api_base=f"http://{base_ip}:{port}",
            retry_strategy=ExponentialBackoffRetryStrategy(max_retries=6),
        )
    elif engine == "anthropic":
        api_key = api_key 
        chat = llms.LiteLLMChat(
            model=f"anthropic/{model}",
            temperature=temperature,
            top_p=top_p,
            retry_strategy=ExponentialBackoffRetryStrategy(max_retries=6),
        )
    elif engine == "gemini":
        api_key = api_key 
        chat = llms.LiteLLMChat(
            model=f"gemini/{model}",
            temperature=temperature,
            top_p=top_p,
            retry_strategy=ExponentialBackoffRetryStrategy(max_retries=6),
        )
    elif engine == "deepseek":
        api_key = api_key 
        chat = llms.LiteLLMChat(
            model=f"deepseek/{model}",
            temperature=temperature,
            top_p=top_p,
            retry_strategy=ExponentialBackoffRetryStrategy(max_retries=6),
        )
    elif engine == "mistral":
        api_key = api_key 
        chat = llms.LiteLLMChat(
            model=f"mistral/{model}",
            temperature=temperature,
            top_p=top_p,
            retry_strategy=ExponentialBackoffRetryStrategy(max_retries=6),
        )
    elif engine == "openai":
        api_key = api_key
        chat = llms.OpenAIChat(
            model=model,
            retry_strategy=FixedDelayRetryStrategy(),
            cache_strategy=DefaultCache(),
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unsupported engine: {engine}")
    
    return chat

"""
parser = argparse.ArgumentParser(description="RAG Server")
parser.add_argument("--base_ip", default="localhost", help="Base IP address for the engine")
parser.add_argument("--rag_port", default=8081, type=int, help="Port for the RAG server")
parser.add_argument("--port", default="11434", help="Port for the engine")
parser.add_argument("--engine", default="ollama", help="Name of the engine to use")
parser.add_argument("--model", default="mistral", help="Name of the model to use")
parser.add_argument("--api_key", default=None, help="API key for the selected engine")
parser.add_argument("--temperature", default=0.7, type=float, help="Temperature setting for the LLM")
parser.add_argument("--top_p", default=0.2, type=float, help="Top P setting for the LLM")

args = parser.parse_args()

threading.Thread(target=run_pathway_pipeline,
                 args=(args.base_ip, args.rag_port,
                       args.port,
                       args.engine,
                       args.model,
                       args.api_key,
                       args.temperature,
                       args.top_p), daemon=True).start()

# Wait for the RAG server to be ready
RAG_READY_EVENT.wait()

print(f"Pathway RAG server started at http://{args.base_ip}:{args.rag_port}")"""