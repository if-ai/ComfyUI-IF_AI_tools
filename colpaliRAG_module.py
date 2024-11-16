import os
import logging
import torch
import builtins
from byaldi import RAGMultiModalModel
from .graphRAG_module import GraphRAGapp
from typing import Tuple, Optional, Dict, Union, List, Any
from pathlib import Path
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from .send_request import send_request
import asyncio
import json
import shutil
from PIL import Image
from io import BytesIO
from pdf2image import convert_from_path
from .utils import get_api_key, load_placeholder_image

import comfy.model_management as mm
from comfy.utils import ProgressBar
import folder_paths
from .transformers_api import TransformersModelManager

import sys

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variable for model caching
if not hasattr(builtins, 'global_colpali_model'):
    builtins.global_colpali_model = None

class colpaliRAGapp:
    def __init__(self):
        self.comfy_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.rag_dir = os.path.join(folder_paths.base_path, "custom_nodes", "ComfyUI-IF_AI_tools", "IF_AI", "rag")
        self._rag_root_dir = None
        self._input_dir = None
        self.graphrag_app = GraphRAGapp()
        self.transformers_api = TransformersModelManager()
        self.cached_index_model = None  # Add this line to store the loaded index
        self.cached_index_name = None   # Add this line to track which index is loaded

    @property
    def rag_root_dir(self):
        return self._rag_root_dir

    @rag_root_dir.setter
    def rag_root_dir(self, value):
        self._rag_root_dir = value
        self._input_dir = os.path.join(self.rag_dir, value, "input") if value else None
        logger.debug(f"rag_root_dir setter: set to {self._rag_root_dir}")
        logger.debug(f"input_dir set to {self._input_dir}")
        if self._input_dir:
            os.makedirs(self._input_dir, exist_ok=True)
            logger.debug(f"Created input directory: {self._input_dir}")

    def set_rag_root_dir(self, rag_folder_name):
        if rag_folder_name:
            new_rag_root_dir = os.path.join(self.rag_dir, rag_folder_name)
        else:
            new_rag_root_dir = os.path.join(self.rag_dir, "rag_data")

        self._rag_root_dir = rag_folder_name  # Ensure rag_folder_name is a string
        self._input_dir = os.path.join(new_rag_root_dir, "input")

        # Ensure directories exist
        os.makedirs(self._rag_root_dir, exist_ok=True)
        os.makedirs(self._input_dir, exist_ok=True)

        logger.debug(f"set_rag_root_dir: rag_root_dir set to {self._rag_root_dir}")
        logger.debug(f"set_rag_root_dir: input_dir set to {self._input_dir}")
        return self._rag_root_dir, self._input_dir

    @classmethod
    def get_colpali_model(cls, query_type):
        logger.debug(f"Attempting to get ColPali model. Current state: {'Loaded' if builtins.global_colpali_model is not None else 'Not loaded'}")

        if query_type == "colqwen2":
            model_path = os.path.join(folder_paths.models_dir, "LLM", "colqwen2-v0.1")
        elif query_type == "colpali-v1.2":
            model_path = os.path.join(folder_paths.models_dir, "LLM", "colpali-v1.2")
        elif query_type == "colpali":
            model_path = os.path.join(folder_paths.models_dir, "LLM", "colpali")
        else:
            logger.error(f"Invalid query type: {query_type}")
            return None

        if builtins.global_colpali_model is None:
            torch.cuda.empty_cache()
            logger.info(f"Loading ColPali model from {model_path}")
            try:
                builtins.global_colpali_model = RAGMultiModalModel.from_pretrained(
                    model_path,
                    device="cuda",
                    verbose=1,
                )
                logger.info("ColPali model loaded and cached globally on CUDA")
            except Exception as e:
                logger.error(f"Error loading ColPali model: {str(e)}")
                builtins.global_colpali_model = None
        else:
            logger.info("Using existing globally cached ColPali model")
        return builtins.global_colpali_model

    async def insert(self):
        try:
            logger.debug("Starting insert function...")
            settings = self.graphrag_app.load_settings()

            index_path = os.path.join(self.rag_dir, settings.get("rag_folder_name"), "input")
            index_name = settings.get("rag_folder_name")
            query_type = settings.get("query_type", "colqwen")

            # Get the model
            colpali_model = self.get_colpali_model(query_type)
            if colpali_model is None:
                logger.error("Failed to load ColPali model for indexing")
                return False

            # Check if index already exists
            index_folder = os.path.join(".byaldi", index_name)
            is_existing_index = os.path.exists(index_folder)

            # Patch the processor's process_images method
            original_process_images = colpali_model.model.processor.process_images

            def process_images_wrapper(*args, **kwargs):
                result = original_process_images(*args, **kwargs)
                processed = {}
                for k, v in result.items():
                    if torch.is_tensor(v):
                        if 'pixel_values' in k:
                            processed[k] = v.to(dtype=torch.bfloat16)
                        else:
                            processed[k] = v.to(dtype=torch.long)
                    else:
                        processed[k] = v
                return processed

            try:
                # Apply the patch
                colpali_model.model.processor.process_images = process_images_wrapper

                # Create new index or add to existing one
                if is_existing_index:
                    logger.info(f"Found existing index: {index_name}, loading it first...")
                    # Load existing index
                    colpali_model = self.get_colpali_model(query_type)
                    colpali_model = colpali_model.from_index(
                        index_name,
                        index_root=".byaldi",
                        device="cuda",
                        verbose=1
                    )

                    # Get list of already indexed files
                    indexed_files = set()
                    if hasattr(colpali_model.model, 'doc_ids_to_file_names'):
                        indexed_files = set(colpali_model.model.doc_ids_to_file_names.values())

                    # Process new files only
                    new_files = []
                    for file in os.listdir(index_path):
                        file_path = os.path.join(index_path, file)
                        if file_path not in indexed_files:
                            new_files.append(file_path)

                    if new_files:
                        logger.info(f"Adding {len(new_files)} new documents to existing index")
                        colpali_model.index(
                            input_path=new_files,
                            index_name=index_name,
                            store_collection_with_index=False,
                            overwrite=False,
                            max_image_width=1024,
                            max_image_height=1024,
                        )
                    else:
                        logger.info("No new documents to add to the index")

                else:
                    logger.info(f"Creating new index: {index_name}")
                    colpali_model.index(
                        input_path=index_path,
                        index_name=index_name,
                        store_collection_with_index=False,
                        overwrite=False,
                        max_image_width=1024,
                        max_image_height=1024,
                    )

                # Remove extra folder if it exists
                extra_folder = os.path.join(self.comfy_dir, settings.get("rag_folder_name"))
                if os.path.exists(extra_folder) and os.path.isdir(extra_folder):
                    logger.debug(f"Removing extra folder: {extra_folder}")
                    shutil.rmtree(extra_folder)

                return True

            finally:
                # Restore original process_images method
                colpali_model.model.processor.process_images = original_process_images

        except Exception as e:
            logger.error(f"Error during indexing: {str(e)}")
            return False

    def load_indexed_images(self, index_name, size):
        """Load previously indexed images from the stored index and convert new PDFs"""
        try:
            index_path = os.path.join(self.rag_dir, index_name, "input")
            images_path = os.path.join(self.rag_dir, index_name, "converted_images")
            os.makedirs(images_path, exist_ok=True)

            # Get sorted list of PDF files to ensure consistent ordering
            pdf_files = sorted([f for f in os.listdir(index_path) if f.lower().endswith('.pdf')])

            if not pdf_files:
                logger.warning("No PDF files found in the input directory.")
                return None

            all_images = {}

            # Process each PDF file
            for doc_id, pdf_file in enumerate(pdf_files):
                pdf_name = os.path.splitext(pdf_file)[0]
                pdf_images_dir = os.path.join(images_path, pdf_name)
                pdf_path = os.path.join(index_path, pdf_file)

                logger.debug(f"Processing PDF {pdf_file} as doc_id {doc_id}")

                # Check if conversion is needed
                needs_conversion = True
                if os.path.exists(pdf_images_dir) and os.listdir(pdf_images_dir):
                    pdf_mtime = os.path.getmtime(pdf_path)
                    newest_image = max(
                        os.path.getmtime(os.path.join(pdf_images_dir, f))
                        for f in os.listdir(pdf_images_dir)
                        if f.endswith('.png')
                    )
                    needs_conversion = pdf_mtime > newest_image

                if needs_conversion:
                    logger.info(f"Converting PDF: {pdf_file}")
                    os.makedirs(pdf_images_dir, exist_ok=True)
                    images = convert_from_path(
                        pdf_path,
                        thread_count=os.cpu_count() - 1,
                        fmt='png',
                        paths_only=False,
                        size=size,
                    )

                    # Save converted images with consistent naming
                    for page_num, img in enumerate(images, 1):
                        img_path = os.path.join(pdf_images_dir, f"page_{page_num:03d}.png")
                        img.save(img_path, "PNG")
                        logger.debug(f"Saved {img_path}")

                # Load images in correct order
                page_files = sorted(
                    [f for f in os.listdir(pdf_images_dir) if f.endswith('.png')],
                    key=lambda x: int(x.split('_')[1].split('.')[0])  # Sort by page number
                )

                images = []
                for page_file in page_files:
                    img_path = os.path.join(pdf_images_dir, page_file)
                    try:
                        img = Image.open(img_path)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        images.append(img)
                        logger.debug(f"Loaded {img_path}")
                    except Exception as e:
                        logger.error(f"Error loading {img_path}: {e}")
                        continue

                all_images[doc_id] = images
                logger.debug(f"Loaded {len(images)} pages for document {doc_id} ({pdf_file})")

            return all_images

        except Exception as e:
            logger.error(f"Error loading indexed images: {str(e)}")
            return None

    def get_top_results(self, results, all_images, index_name, llm_provider="transformers"):
        """
        Get relevant images and their metadata based on search results.
        Returns lists of images and masks along with result information.
        
        Args:
            results: Search results to process
            all_images: Dictionary of loaded images
            index_name: Name of the index being used
            llm_provider: The LLM provider being used (default: "transformers")
        """
        top_results_images = []
        top_results_masks = []
        result_info = []

        try:
            # Get list of PDF files
            index_path = os.path.join(self.rag_dir, index_name, "input")
            pdf_files = sorted([f for f in os.listdir(index_path) if f.lower().endswith('.pdf')])

            # Sort and filter results with safe score extraction
            def get_score(result):
                try:
                    return float(result.score) if hasattr(result, 'score') else 0.0
                except (ValueError, TypeError):
                    return 0.0

            sorted_results = sorted(results, key=get_score, reverse=True)

            # For non-standard LLM providers, only take the highest scoring result
            if llm_provider.lower() not in ["transformers", "openai", "anthropic"]:
                sorted_results = sorted_results[:1]

            logger.debug("Processing sorted results:")
            for r in sorted_results:
                score = get_score(r)
                try:
                    doc_id = int(r.doc_id) if isinstance(r.doc_id, str) else r.doc_id
                    page_num = int(r.page_num) if isinstance(r.page_num, str) else r.page_num
                    logger.debug(f"Doc: {doc_id}, Page: {page_num}, Score: {score}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid result format: {e}")
                    continue

            # Create a mapping of original doc_ids to ensure correct ordering
            doc_id_map = {i: os.path.splitext(pdf_file)[0] for i, pdf_file in enumerate(pdf_files)}

            for result in sorted_results:
                # Updated integer conversion logic
                try:
                    doc_id = int(result.doc_id) if isinstance(result.doc_id, str) else result.doc_id
                    page_num = int(result.page_num) if isinstance(result.page_num, str) else result.page_num
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid doc_id or page_num format: {e}")
                    continue

                # Validate document ID
                if doc_id not in all_images:
                    logger.warning(f"Document ID {doc_id} not found in loaded images")
                    continue

                # Validate page number (convert to 0-based index)
                page_idx = page_num - 1
                if page_idx < 0 or page_idx >= len(all_images[doc_id]):
                    logger.warning(f"Invalid page {page_num} for document {doc_id}")
                    continue

                # Get image for this result
                try:
                    image = all_images[doc_id][page_idx]
                    logger.debug(f"Retrieved image for doc {doc_id} ('{doc_id_map[doc_id]}'), page {page_num}")

                    # Ensure image is in RGB format
                    if image.mode != 'RGB':
                        image = image.convert('RGB')

                    # Convert to tensor
                    img_array = np.array(image).astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_array)[None,]

                    # Create corresponding mask
                    mask_tensor = torch.ones((1, img_array.shape[0], img_array.shape[1]), 
                                        dtype=torch.float32, device="cpu")

                    # Add to results
                    top_results_images.append(img_tensor)
                    top_results_masks.append(mask_tensor)

                    # Store result info
                    result_info.append({
                        "doc_id": doc_id,
                        "page_num": page_num,
                        "score": float(result.score) if hasattr(result, 'score') else 0.0,
                        "pdf_name": doc_id_map[doc_id],
                        "metadata": result.metadata if hasattr(result, 'metadata') else {}
                    })

                except Exception as e:
                    logger.error(f"Error processing image for doc {doc_id}, page {page_num}: {e}")
                    continue

            if top_results_images:
                # Combine tensors
                try:
                    top_combined_images = torch.cat(top_results_images, dim=0)
                    top_combined_masks = torch.cat(top_results_masks, dim=0)

                    # Add debug logging for tensor shapes
                    logger.debug(f"Number of images processed: {len(top_results_images)}")
                    logger.debug(f"Combined images tensor shape: {top_combined_images.shape}")
                    logger.debug(f"Combined masks tensor shape: {top_combined_masks.shape}")
                    
                    if llm_provider.lower() not in ["transformers", "openai", "anthropic"]:
                        # Verify we only have one image
                        assert top_combined_images.shape[0] == 1, "Expected only one image for non-standard LLM provider"
                        assert top_combined_masks.shape[0] == 1, "Expected only one mask for non-standard LLM provider"
                        logger.debug("Confirmed single image output for non-standard LLM provider")

                    logger.debug("Final result order:")
                    for info in result_info:
                        logger.debug(f"Doc: {info['doc_id']} ({info['pdf_name']}), Page: {info['page_num']}, Score: {info['score']:.2f}")

                    return top_combined_images, top_combined_masks, result_info

                except Exception as e:
                    logger.error(f"Error combining tensors: {e}")
                    return None, None, []

            logger.debug("No valid images to process")
            return None, None, []

        except Exception as e:
            logger.error(f"Error in get_top_results: {str(e)}")
            return None, None, []

    async def query(self, prompt: str, query_type: str, system_message_str: str, **kwargs):
        try:
            # 1. Initialize settings and parameters
            settings: dict = self.graphrag_app.load_settings()
            llm_provider: str = kwargs.pop("llm_provider", settings.get("llm_provider", "ollama"))
            base_ip: str = kwargs.pop("base_ip", settings.get("base_ip", "localhost"))
            port: str = kwargs.pop("port", settings.get("port", "11434"))
            llm_model: str = kwargs.pop('llm_model', settings.get("llm_model", "llama3.1:latest"))
            llm_api_key: str = settings.get('external_llm_api_key') if settings.get('external_llm_api_key') != "" else get_api_key(f"{settings['llm_provider'].upper()}_API_KEY", settings['llm_provider'])
            keep_alive: str = kwargs.pop("keep_alive", settings.get("keep_alive", "False"))
            seed: str = kwargs.pop("seed", settings.get("seed", "None"))
            temperature: float = float(kwargs.pop("temperature", settings.get("temperature", "0.7")))
            top_p: float = float(kwargs.pop("top_p", settings.get("top_p", "0.90")))
            top_k: int = int(kwargs.pop("top_k", settings.get("top_k", "40")))
            max_tokens: int = int(kwargs.pop("max_tokens", settings.get("max_tokens", "2048")))
            presence_penalty: float = float(kwargs.pop("repeat_penalty", settings.get("repeat_penalty", "1.2")))
            random: str = kwargs.pop("random", settings.get("random", "False"))
            stop: str = kwargs.pop("stop", settings.get("stop", "None"))
            precision: str = kwargs.pop("precision", settings.get("precision", "fp16"))
            attention: str = kwargs.pop("attention", settings.get("attention", "sdpa"))
            index_name: str = kwargs.pop("rag_folder_name", settings.get("rag_folder_name"))
            prime_directives: str = kwargs.pop("prime_directives", settings.get("prime_directives", "None"))
            aspect_ratio: str = kwargs.pop("aspect_ratio", settings.get("aspect_ratio", "16:9"))
            top_k_search: str = kwargs.pop("top_k_search", settings.get("top_k_search", "3"))
            #vertical/horizontal is x/y on pdf2image that is why I inverted the aspect ratio
            size: tuple[int, int] = (768, 1024) if aspect_ratio == "16:9" else (1024, 768) if aspect_ratio == "9:16" else (1024, 1024)
            messages: list = []
            if prime_directives != "None":
                system_message_str = prime_directives
            elif system_message_str == "None":
                system_message_str = "You are a helpful assistant. Analyze the image and answer the user's question."

            # 2. Get and validate model
            colpali_model = await self._prepare_model(query_type, index_name)
            if not colpali_model:
                return self._create_error_response(prompt, "ColPali model not available")

            # 3. Perform search and validate results
            top_k_search_int: int = int(top_k_search)
            results: Union[List[str], List[Any]] = await self._perform_search(
                colpali_model, 
                str(prompt), 
                top_k_search_int,
                llm_provider=llm_provider
            )
            if not results:
                return self._create_error_response(prompt, "No relevant documents found", tool_output="Search returned no results")

            # 4. Process images
            image_data: Optional[Union[
                Tuple[torch.Tensor, torch.Tensor, List[Dict]],
                Tuple[torch.Tensor, torch.Tensor, List[str]]
            ]] = await self._process_images(
                results, 
                index_name, 
                size,
                llm_provider=llm_provider
            )
            if not image_data:
                return self._create_error_response(prompt, "Failed to process images")

            images_tensor: torch.Tensor
            masks_tensor: torch.Tensor
            result_info: Union[List[Dict], List[str]]
            images_tensor, masks_tensor, result_info = image_data       

            logger.debug(f"Images tensor shape: {images_tensor.shape}")
            logger.debug(f"Result info: {result_info}")

            try:
                generated_text: Union[str, List[str]]
                generated_text = await send_request(
                    llm_provider=llm_provider,
                    base_ip=base_ip,
                    port=port,
                    images=images_tensor,
                    llm_model=llm_model,
                    system_message=system_message_str,
                    user_message=prompt,
                    messages=messages,
                    seed=seed,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    random=random,
                    top_k=top_k,
                    top_p=top_p,
                    repeat_penalty=presence_penalty,
                    stop=stop,
                    keep_alive=keep_alive,
                    llm_api_key=llm_api_key if llm_api_key != "" else None,
                    precision=precision,
                    attention=attention,
                )

                # Handle case where generated_text is a list
                if isinstance(generated_text, list):
                    generated_text = "\n".join(generated_text)  # Join with newlines to preserve formatting

            except Exception as e:
                logger.error(f"Error in API request: {str(e)}")
                return {
                    "Question": prompt,
                    "Response": f"Error communicating with {llm_provider}: {str(e)}",
                    "Negative": "",
                    "Tool_Output": str(result_info),
                    "Retrieved_Image": images_tensor.detach() if torch.is_tensor(images_tensor) else None,
                    "Mask": masks_tensor.detach() if torch.is_tensor(masks_tensor) else None
                }

            # 6. Format and return response
            return {
                "Question": prompt,
                "Response": generated_text,
                "Negative": "",
                "Tool_Output": self._format_tool_output(result_info),
                "Retrieved_Image": images_tensor,
                "Mask": masks_tensor
            }

        except Exception as e:
            logger.error(f"Error in colpali query: {str(e)}")
            return self._create_error_response(prompt, f"Error processing query: {str(e)}")

    async def _prepare_model(self, query_type, index_name):
        """Prepare and validate the ColPali model"""
        try:
            # Check if we already have the correct index loaded
            if self.cached_index_model is not None and self.cached_index_name == index_name:
                logger.debug(f"Using cached index model for {index_name}")
                return self.cached_index_model

            # Get base model
            colpali_model = self.get_colpali_model(query_type)
            if not colpali_model:
                logger.error("Failed to get base ColPali model")
                return None

            # Load new index
            logger.debug(f"Loading new index {index_name} from .byaldi...")
            try:
                model = RAGMultiModalModel.from_index(
                    index_name,
                    index_root=".byaldi",
                    device="cuda",
                    verbose=1
                )

                # Verify index loaded correctly
                if not hasattr(model.model, 'indexed_embeddings') or not model.model.indexed_embeddings:
                    logger.error("Index loaded but no embeddings found")
                    return None

                # Cache the successfully loaded index
                self.cached_index_model = model
                self.cached_index_name = index_name

                logger.debug(f"Successfully loaded and cached index with {len(model.model.indexed_embeddings)} embeddings")
                return model

            except Exception as e:
                logger.error(f"Error loading index: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"Error in _prepare_model: {str(e)}")
            return None

    async def _perform_search(self, model, prompt, top_k_search, llm_provider="transformers"):
        """
        Perform search and validate results
        
        Args:
            model: The RAG model to use for search
            prompt: The search prompt
            top_k_search: Number of results to return
            llm_provider: The LLM provider being used (default: "transformers")
        """
        logger.debug(f"Searching with prompt: {prompt}")
        
        # Adjust top_k based on llm_provider
        if llm_provider.lower() not in ["transformers", "openai", "anthropic"]:
            top_k_search = 1
            
        results = model.search(prompt, k=top_k_search)

        if results:
            for result in results:
                if hasattr(result, 'doc_id'):
                    result.doc_id = int(result.doc_id) if isinstance(result.doc_id, str) else result.doc_id
                if hasattr(result, 'page_num'):
                    result.page_num = int(result.page_num) if isinstance(result.page_num, str) else result.page_num
        return results

    async def _process_images(self, results, index_name, size, llm_provider="transformers"):
        """Process and validate images from search results"""
        all_images = self.load_indexed_images(index_name, size)
        if all_images is None:
            return None

        return self.get_top_results(results, all_images, index_name, llm_provider)

    def _format_tool_output(self, result_info):
        """Format tool output text"""
        tool_text = "Retrieved Documents:\n"
        for info in result_info:
            tool_text += f"\nDocument {info['doc_id']}, Page {info['page_num']}"
            tool_text += f"\nRelevance Score: {info['score']:.2f}"
            if info['metadata']:
                tool_text += f"\nMetadata: {info['metadata']}"
            tool_text += "\n"
        return tool_text

    def _create_error_response(self, prompt, error_message, tool_output=None):
        """Create standardized error response"""
        return {
            "Question": prompt,
            "Response": error_message,
            "Negative": "",
            "Tool_Output": tool_output,
            "Retrieved_Image": None,
            "Mask": None
        }

    def cleanup(self):
        """Cleanup method to free up GPU memory when needed"""
        if builtins.global_colpali_model:
            del builtins.global_colpali_model
            builtins.global_colpali_model = None
        torch.cuda.empty_cache()
        logger.info("Cleaned up models and freed GPU memory")

    def cleanup_index(self):
        """Method to manually clear the cached index if needed"""
        self.cached_index_model = None
        self.cached_index_name = None
        torch.cuda.empty_cache()
        logger.info("Cleared cached index and freed GPU memory")

    async def _generate_text_response(self, images_tensor, prompt, system_message_str, params):
        """Generate text response based on provider"""
