import torch
import torchvision.transforms.functional as F
import numpy as np
import logging
from PIL import Image, ImageDraw, ImageFont, ImageColor
import supervision as sv
from io import BytesIO
import base64
import json
import random
import os
import re
import comfy.model_management as mm
from .transformers_api import TransformersModelManager
from torchvision.transforms import functional as TF
from supervision.detection.lmm import from_florence_2
from json import JSONEncoder
from typing import Tuple, Optional, List, Union
import folder_paths

logger = logging.getLogger(__name__)

class NumpyEncoder(JSONEncoder):
    """Custom JSON Encoder that handles NumPy arrays and torch tensors."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return super().default(obj)

SUPPORTED_TASKS_FLORENCE_2 = [
    "<OD>",
    "<CAPTION_TO_PHRASE_GROUNDING>",
    "<DENSE_REGION_CAPTION>",
    "<REGION_PROPOSAL>",
    "<OCR_WITH_REGION>",
    "<REFERRING_EXPRESSION_SEGMENTATION>",
    "<REGION_TO_SEGMENTATION>",
    "<OPEN_VOCABULARY_DETECTION>",
    "<REGION_TO_CATEGORY>",
    "<REGION_TO_DESCRIPTION>"
]

def process_mask(mask, image_size=None):
    """Process mask to ensure compatibility with ComfyUI."""
    if mask is None:
        return None
        
    # Convert to numpy if tensor
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # Handle boolean masks
    if mask.dtype == bool:
        mask = mask.astype(np.float32)
    
    # Ensure float32
    mask = mask.astype(np.float32)
    
    # Convert to tensor
    mask = torch.from_numpy(mask)
    
    # Handle different shapes
    if len(mask.shape) == 2:  # Single mask
        mask = mask.unsqueeze(0)
    elif len(mask.shape) == 3:  # Multiple masks
        if mask.shape[0] > 1:  # Multiple masks to combine
            if image_size is not None:  # Resize individual masks if needed
                W, H = image_size
                resized_masks = []
                for m in mask:
                    if m.shape != (H, W):
                        m = F.interpolate(
                            m.unsqueeze(0).unsqueeze(0),
                            size=(H, W),
                            mode='nearest'
                        ).squeeze()
                    resized_masks.append(m)
                mask = torch.stack(resized_masks)
            # Combine masks if multiple
            mask = mask.any(dim=0).float().unsqueeze(0)
    
    # Final resize if needed
    if image_size is not None:
        W, H = image_size
        if mask.shape[-2:] != (H, W):
            mask = F.interpolate(
                mask.unsqueeze(0),
                size=(H, W),
                mode='nearest'
            ).squeeze(0)
    
    return mask

def process_mask_selection(masks, selection, labels=None):
    """Process mask selection based on indices or labels."""
    if not selection or masks is None:
        return masks
        
    selections = selection.split(',')
    mask_indices = []
    
    for sel in selections:
        sel = sel.strip()
        if sel.isdigit():
            idx = int(sel)
            if 0 <= idx < len(masks):
                mask_indices.append(idx)
        elif labels is not None:
            for i, label in enumerate(labels):
                if sel.lower() in label.lower():
                    mask_indices.append(i)
    
    if not mask_indices:
        return masks
        
    selected_masks = masks[mask_indices]
    return selected_masks.any(dim=0).float().unsqueeze(0)

def generate_mask_from_box(box: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """
    Generate a binary mask from a bounding box.
    
    Args:
        box (np.ndarray): Array of [x1, y1, x2, y2] coordinates
        image_size (Tuple[int, int]): (width, height) of the image
        
    Returns:
        np.ndarray: Binary mask array of shape (H, W)
    """
    W, H = image_size
    mask = np.zeros((H, W), dtype=np.bool_)
    x1, y1, x2, y2 = map(lambda x: max(0, int(x)), box)  # Ensure non-negative integers
    x2 = min(x2, W)  # Ensure within image bounds
    y2 = min(y2, H)
    if x2 > x1 and y2 > y1:  # Only set mask if box is valid
        mask[y1:y2, x1:x2] = True
    return mask



class FlorenceModule:
    def __init__(self):
        self.model_manager = TransformersModelManager()
        self.device = self.model_manager.device
        self.offload_device = self.model_manager.offload_device
        self.comfy_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.placeholder_image_path = os.path.join(folder_paths.base_path, "custom_nodes",  "ComfyUI-IF_AI_tools", "IF_AI", "placeholder.png")

        self.box_annotator = sv.BoxAnnotator(
            color=sv.ColorPalette.DEFAULT,
            thickness=2,
            color_lookup=sv.ColorLookup.CLASS
        )
        self.label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.DEFAULT,
            text_color=sv.Color.WHITE,
            text_scale=0.5,
            text_thickness=1,
            text_padding=10,
            text_position=sv.Position.TOP_LEFT,
            color_lookup=sv.ColorLookup.CLASS
        )
        self.mask_annotator = sv.MaskAnnotator(
            color=sv.ColorPalette.DEFAULT,
            opacity=0.5,
            color_lookup=sv.ColorLookup.CLASS
        )

        self.colormap = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 
                        'olive', 'cyan', 'red', 'lime', 'indigo', 'violet', 
                        'aqua', 'magenta', 'gold', 'tan', 'skyblue']

    def prepare_json_output(self, out_data):
        """Prepare output data for JSON serialization."""
        try:
            return json.dumps(out_data, cls=NumpyEncoder, indent=2)
        except Exception as e:
            logger.error(f"Error serializing output data: {e}")
        return json.dumps({"error": str(e)})

    def format_output_data(self, detections, labels, mask, W, H, task):
        """Format detection data for output."""
        try:
            output = {
                "boxes": detections.xyxy.tolist() if detections.xyxy is not None else [],
                "task": task,
                "dimensions": {"width": W, "height": H}
            }
            
            if labels is not None:
                if isinstance(labels, (np.ndarray, torch.Tensor)):
                    output["labels"] = labels.tolist()
                else:
                    output["labels"] = list(labels)
            else:
                output["labels"] = []

            if mask is not None:
                if isinstance(mask, (np.ndarray, torch.Tensor)):
                    output["has_mask"] = True
                    output["mask_shape"] = list(mask.shape)
                else:
                    output["has_mask"] = False
            else:
                output["has_mask"] = False

            return output
        except Exception as e:
            logger.error(f"Error formatting output data: {e}")
            return {"error": str(e)}
        
    def parse_florence_response(self, text):
        """Parse Florence response to extract labels and locations."""
        pattern = r'([^<]+)(?:<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>)'
        matches = re.finditer(pattern, text)
        
        labels = []
        locations = []
        
        for match in matches:
            label = match.group(1).strip()
            coords = [int(match.group(i)) for i in range(2, 6)]
            labels.append(label)
            locations.append(coords)
            
        return labels, np.array(locations) if locations else np.array([])

    def validate_task(self, task_prompt: str) -> str:
        """Validate and format task prompt."""
        task_key = f"<{task_prompt.upper()}>" if not task_prompt.startswith("<") else task_prompt
        if task_key not in SUPPORTED_TASKS_FLORENCE_2:
            raise ValueError(f"Task {task_key} not supported. Supported tasks are: {SUPPORTED_TASKS_FLORENCE_2}")
        return task_key

    def handle_task_specific_processing(self, task: str, response, W: int, H: int):
        """Handle task-specific processing for Florence output."""
        xyxy, labels, mask, xyxyxyxy = from_florence_2(response, (W, H))
        
        # Task-specific processing
        if task in ["<REFERRING_EXPRESSION_SEGMENTATION>", "<REGION_TO_SEGMENTATION>"]:
            # These tasks return masks directly
            if mask is None:
                logger.warning(f"No mask returned for segmentation task {task}")
                return xyxy, labels, mask, xyxyxyxy
            
        elif task in ["<OD>", "<CAPTION_TO_PHRASE_GROUNDING>", "<DENSE_REGION_CAPTION>"]:
            # These tasks return boxes and labels
            if xyxy is None or len(xyxy) == 0:
                logger.warning(f"No boxes returned for detection task {task}")
                return xyxy, labels, mask, xyxyxyxy
                
            # Generate masks from boxes
            if mask is None:
                try:
                    masks_list = []
                    for box in xyxy:
                        box_mask = generate_mask_from_box(box, (W, H))
                        if box_mask is not None:
                            masks_list.append(box_mask)
                    mask = np.stack(masks_list) if masks_list else None
                    logger.debug(f"Generated {len(masks_list)} masks from boxes for {task}")
                except Exception as e:
                    logger.error(f"Error generating masks from boxes: {e}")
                    mask = None
                    
        elif task == "<OCR_WITH_REGION>":
            # Handle OCR with special oriented boxes
            if xyxyxyxy is not None:
                logger.debug(f"Processing OCR with oriented boxes: {len(xyxyxyxy)} regions")
                
        elif task in ["<REGION_TO_CATEGORY>", "<REGION_TO_DESCRIPTION>"]:
            # These tasks return a single region with description
            if labels is not None and len(labels) > 0:
                logger.debug(f"Region description: {labels[0]}")
                
        return xyxy, labels, mask, xyxyxyxy

    async def run_florence(self, images, task, task_prompt, llm_model, precision, attention,
                        fill_mask, output_mask_select, keep_alive, max_new_tokens,
                        temperature, top_p, top_k, repetition_penalty, seed, text_input):
        try:
            # Validate task and format prompt
            task_key = self.validate_task(task_prompt)
            prompt = f"{task_key} {text_input}" if text_input else task_key
            logger.debug(f"Using task: {task_key} with prompt: {prompt}")

            if len(images.shape) == 3:
                images = images.unsqueeze(0)
            images = images.permute(0, 3, 1, 2)
            
            out = []
            out_masks = []
            out_results = []
            out_data = []

            for img in images:
                try:
                    image_pil = TF.to_pil_image(img)
                    W, H = image_pil.size

                    result = await self.model_manager.send_transformers_request(
                        model_name=llm_model,
                        system_message="",
                        user_message=prompt,
                        messages=[],
                        max_new_tokens=max_new_tokens,
                        images=[image_pil],
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        stop_strings_list=["<|endoftext|>"],
                        repetition_penalty=repetition_penalty,
                        seed=seed,
                        keep_alive=keep_alive,
                        precision=precision,
                        attention=attention
                    )

                    generated_text = result[0]
                    response = result[1][0] if isinstance(result[1], list) else result[1]
                    
                    # Process Florence output with task-specific handling
                    xyxy, labels, mask, xyxyxyxy = self.handle_task_specific_processing(
                        task_key, response, W, H
                    )

                    # Generate masks for bounding boxes if no mask was provided
                    if mask is None and xyxy is not None and len(xyxy) > 0:
                        try:
                            masks_list = []
                            for box in xyxy:
                                box_mask = generate_mask_from_box(box, (W, H))
                                if box_mask is not None:
                                    masks_list.append(box_mask)
                            mask = np.stack(masks_list) if masks_list else None
                            logger.debug(f"Generated {len(masks_list)} masks from boxes")
                        except Exception as e:
                            logger.error(f"Error generating masks from boxes: {e}")
                            mask = None

                    # Create detections object
                    detections = sv.Detections(
                        xyxy=xyxy,
                        mask=mask,
                        class_id=np.arange(len(labels)) if labels is not None else None,
                        data={"class_name": labels} if labels is not None else None
                    )

                    # Create annotated image
                    annotated_frame = np.array(image_pil)

                    # Process and apply masks
                    if mask is not None:
                        # Handle mask selection
                        if output_mask_select:
                            selected_indices = []
                            selections = output_mask_select.split(',')
                            
                            for sel in selections:
                                sel = sel.strip().lower()
                                # Check for label match
                                if labels is not None:
                                    for idx, label in enumerate(labels):
                                        if sel in label.lower():
                                            selected_indices.append(idx)
                                # Check for numeric index
                                elif sel.isdigit():
                                    idx = int(sel)
                                    if 0 <= idx < len(mask):
                                        selected_indices.append(idx)
                            
                            if selected_indices:
                                selected_mask = np.zeros_like(mask[0])
                                for idx in selected_indices:
                                    selected_mask = np.logical_or(selected_mask, mask[idx])
                                mask = np.array([selected_mask])

                        if fill_mask:
                            detections.mask = mask
                            annotated_frame = self.mask_annotator.annotate(
                                scene=annotated_frame,
                                detections=detections
                            )
                            
                            # Convert mask for output
                            processed_mask = process_mask(mask, (W, H))
                            if processed_mask is not None:
                                out_masks.append(processed_mask)

                    # Draw boxes and labels
                    annotated_frame = self.box_annotator.annotate(
                        scene=annotated_frame,
                        detections=detections
                    )

                    if labels is not None:
                        formatted_labels = []
                        for idx, label in enumerate(labels):
                            if output_mask_select:
                                if str(idx) in output_mask_select.split(",") or \
                                   any(sel.lower() in label.lower() for sel in output_mask_select.split(",")):
                                    formatted_labels.append(f"[{idx}] {label}")
                                else:
                                    formatted_labels.append(label)
                            else:
                                formatted_labels.append(f"{label}")

                        annotated_frame = self.label_annotator.annotate(
                            scene=annotated_frame,
                            detections=detections,
                            labels=formatted_labels
                        )

                    # Convert to tensor
                    annotated_frame = Image.fromarray(annotated_frame)
                    out_tensor = TF.to_tensor(annotated_frame).unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
                    out.append(out_tensor)

                    # Store results
                    out_results.append(generated_text)
                    out_data.append(self.format_output_data(detections, labels, mask, W, H, task))

                except Exception as e:
                    logger.error(f"Error processing image: {str(e)}")
                    continue

            # Combine outputs
            if len(out) > 0:
                out_tensor = torch.cat(out, dim=0)
                if len(out_masks) > 0:
                    masks_tensor = torch.cat(out_masks, dim=0)
                else:
                    masks_tensor = torch.zeros((1, out_tensor.shape[1], out_tensor.shape[2]), dtype=torch.float32)
            else:
                out_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                masks_tensor = torch.zeros((1, 64, 64), dtype=torch.float32)

            if not keep_alive:
                self.model_manager.unload_model(llm_model)

            return {
                "Question": text_input,
                "Response": out_results[0] if len(out_results) == 1 else out_results,
                "Negative": "",
                "Tool_Output": self.prepare_json_output(out_data),
                "Retrieved_Image": out_tensor,
                "Mask": masks_tensor
            }

        except Exception as e:
            logger.error(f"Error in run_florence: {str(e)}", exc_info=True)
            # Return valid tensors even in case of error
            error_data = {"error": str(e)}
            return {
                "Question": text_input,
                "Response": f"Error: {str(e)}",
                "Negative": "",
                "Tool_Output": json.dumps(error_data),
                "Retrieved_Image": torch.zeros((1, 64, 64, 3), dtype=torch.float32),
                "Mask": torch.zeros((1, 64, 64), dtype=torch.float32)
            }