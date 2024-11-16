# IFLoadImagesNode.py
import os
import re
import torch
import glob
import hashlib
import logging
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import folder_paths
import shutil
from typing import Tuple, List, Dict, Optional
from server import PromptServer
from aiohttp import web
import json

logger = logging.getLogger(__name__)

def numerical_sort_key(path):
    """Sort file paths by numerical order in filenames"""
    parts = re.split('([0-9]+)', os.path.basename(path))
    parts[1::2] = map(int, parts[1::2])  # Convert number parts to integers
    return parts

class ImageManager:
    THUMBNAIL_PREFIX = "thb_"
    PATH_SEPARATOR = "___"  
    SUBFOLDER_PREFIX = "dir"  
    LEVEL_SEPARATOR = "--"   
    THUMBNAIL_PREFIX = "thb_"
    THUMBNAIL_SIZE = (300, 300)

    VALID_EXTENSIONS = {
        "none": {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"},
        "png": {".png"},
        "jpg": {".jpg", ".jpeg"},
        "webp": {".webp"},
        "gif": {".gif"},
        "bmp": {".bmp"}
    }
    
    @staticmethod
    def sanitize_path_component(component: str) -> str:
        """Sanitize path components for safe filename use"""
        # Replace problematic characters but maintain readability
        sanitized = re.sub(r'[\\/:*?"<>|]', '_', component)
        return sanitized.strip()

    @staticmethod
    def encode_path_to_filename(original_path: str, base_path: str) -> str:
        """
        Convert a full file path to an encoded thumbnail filename that preserves hierarchy
        Format: thb_dir--level1--level2--level3___filename.ext
        """
        try:
            # Normalize paths
            original_path = os.path.abspath(original_path)
            base_path = os.path.abspath(base_path)
            
            # Get relative path components
            rel_path = os.path.relpath(original_path, base_path)
            dir_path = os.path.dirname(rel_path)
            filename = os.path.basename(rel_path)
            
            if dir_path and dir_path != '.':
                # Split directory path and sanitize each component
                dir_parts = [ImageManager.sanitize_path_component(p) 
                           for p in dir_path.split(os.sep)]
                # Create encoded directory string
                dir_encoded = (f"{ImageManager.SUBFOLDER_PREFIX}"
                             f"{ImageManager.LEVEL_SEPARATOR}"
                             f"{ImageManager.LEVEL_SEPARATOR.join(dir_parts)}")
                # Combine with filename
                return f"{ImageManager.THUMBNAIL_PREFIX}{dir_encoded}{ImageManager.PATH_SEPARATOR}{filename}"
            else:
                # No subdirectories
                return f"{ImageManager.THUMBNAIL_PREFIX}{filename}"
                
        except Exception as e:
            logger.error(f"Error encoding path {original_path}: {e}")
            return f"{ImageManager.THUMBNAIL_PREFIX}{os.path.basename(original_path)}"

    @staticmethod
    def decode_thumbnail_name(thumbnail_name: str) -> Tuple[List[str], str]:
        """
        Decode a thumbnail filename back into path components and original filename
        Returns (path_components, filename)
        """
        if not thumbnail_name.startswith(ImageManager.THUMBNAIL_PREFIX):
            return [], thumbnail_name
            
        # Remove prefix
        name_without_prefix = thumbnail_name[len(ImageManager.THUMBNAIL_PREFIX):]
        
        # Split into directory part and filename
        parts = name_without_prefix.split(ImageManager.PATH_SEPARATOR)
        
        if len(parts) == 2:
            # Has directory information
            dir_part, filename = parts
            if dir_part.startswith(ImageManager.SUBFOLDER_PREFIX):
                # Extract directory levels
                dir_levels = dir_part[len(ImageManager.SUBFOLDER_PREFIX):].split(ImageManager.LEVEL_SEPARATOR)
                # Remove empty strings
                dir_levels = [level for level in dir_levels if level]
                return dir_levels, filename
        
        # No directory information
        return [], parts[-1]

    @staticmethod
    def get_original_path(thumbnail_name: str, base_path: str) -> str:
        """Convert a thumbnail name back to its original file path"""
        try:
            dir_levels, filename = ImageManager.decode_thumbnail_name(thumbnail_name)
            
            if dir_levels:
                # Reconstruct path with proper system separators
                subpath = os.path.join(*dir_levels) if dir_levels else ""
                return os.path.normpath(os.path.join(base_path, subpath, filename))
            else:
                return os.path.normpath(os.path.join(base_path, filename))
        except Exception as e:
            logger.error(f"Error decoding thumbnail name {thumbnail_name}: {e}")
            return os.path.join(base_path, thumbnail_name.replace(ImageManager.THUMBNAIL_PREFIX, ""))

    @staticmethod
    def normalize_path(path: str) -> str:
        """Normalize path separators to system format"""
        return os.path.normpath(path.replace('\\', os.sep).replace('/', os.sep))

    @staticmethod
    def get_relative_path(file_path: str, base_path: str) -> str:
        """Get the relative path preserving all folder levels"""
        try:
            return os.path.relpath(file_path, base_path)
        except ValueError:
            # Handle case where paths are on different drives
            return file_path

    @staticmethod
    def get_image_files(folder_path: str, include_subfolders: bool, filter_type: str) -> List[str]:
        """Get list of image files with complete path hierarchy"""
        valid_exts = ImageManager.VALID_EXTENSIONS.get(filter_type.lower(), ImageManager.VALID_EXTENSIONS["none"])
        found_files = []
        
        # Normalize the base folder path
        folder_path = ImageManager.normalize_path(folder_path)

        def is_valid_image(filename: str) -> bool:
            return any(filename.lower().endswith(ext) for ext in valid_exts)

        try:
            if include_subfolders:
                for root, _, filenames in os.walk(folder_path):
                    for filename in filenames:
                        if is_valid_image(filename):
                            full_path = os.path.join(root, filename)
                            # Store absolute paths for consistent handling
                            found_files.append(os.path.abspath(full_path))
            else:
                with os.scandir(folder_path) as entries:
                    for entry in entries:
                        if entry.is_file() and is_valid_image(entry.name):
                            found_files.append(os.path.abspath(entry.path))

            return found_files
        except Exception as e:
            logger.error(f"Error getting image files from {folder_path}: {e}")
            return []

    @staticmethod
    def encode_path_to_filename(original_path: str, base_path: str) -> str:
        """Convert a full file path to an encoded thumbnail filename preserving complete hierarchy"""
        try:
            # Normalize both paths
            original_path = ImageManager.normalize_path(original_path)
            base_path = ImageManager.normalize_path(base_path)
            
            # Get relative path from base_path
            rel_path = ImageManager.get_relative_path(original_path, base_path)
            
            # Split path into components
            path_parts = rel_path.split(os.sep)
            
            if len(path_parts) > 1:
                # Join all directory parts with PATH_SEPARATOR
                dirs = ImageManager.PATH_SEPARATOR.join(path_parts[:-1])
                filename = path_parts[-1]
                return f"{ImageManager.THUMBNAIL_PREFIX}{dirs}{ImageManager.PATH_SEPARATOR}{filename}"
            else:
                # No subdirectories
                return f"{ImageManager.THUMBNAIL_PREFIX}{rel_path}"
        except Exception as e:
            logger.error(f"Error encoding path {original_path}: {e}")
            return f"{ImageManager.THUMBNAIL_PREFIX}{os.path.basename(original_path)}"

    @staticmethod
    def decode_thumbnail_name(thumbnail_name: str) -> Tuple[List[str], str]:
        """Decode a thumbnail filename back into path components and original filename"""
        if not thumbnail_name.startswith(ImageManager.THUMBNAIL_PREFIX):
            return [], thumbnail_name
            
        # Remove prefix
        name_without_prefix = thumbnail_name[len(ImageManager.THUMBNAIL_PREFIX):]
        
        # Split by path separator and handle escaped separators
        parts = name_without_prefix.split(ImageManager.PATH_SEPARATOR)
        
        # Last part is the filename, everything else is path components
        return parts[:-1], parts[-1]

    @staticmethod 
    def get_original_path(thumbnail_name: str, base_path: str) -> str:
        """Convert a thumbnail name back to its original file path"""
        try:
            path_parts, filename = ImageManager.decode_thumbnail_name(thumbnail_name)
            
            # Normalize base path
            base_path = ImageManager.normalize_path(base_path)
            
            if path_parts:
                # Reconstruct path with proper system separators
                subpath = os.path.join(*path_parts)
                return os.path.normpath(os.path.join(base_path, subpath, filename))
            else:
                return os.path.normpath(os.path.join(base_path, filename))
        except Exception as e:
            logger.error(f"Error decoding thumbnail name {thumbnail_name}: {e}")
            return os.path.join(base_path, thumbnail_name.replace(ImageManager.THUMBNAIL_PREFIX, ""))
    
    @staticmethod
    def create_thumbnails(folder_path: str, include_subfolders: bool = True,
                         filter_type: str = "none", sort_method: str = "alphabetical",
                         start_index: int = 0, max_images: Optional[int] = None) -> Tuple[bool, str, List[str], Dict[str, int]]:
        try:
            input_dir = folder_paths.get_input_directory()
            thumbnail_paths = []
            image_order = {}  # Track image order

            # Normalize paths
            if not os.path.isabs(folder_path):
                folder_path = os.path.abspath(os.path.join(folder_paths.get_input_directory(), folder_path))
            folder_path = ImageManager.normalize_path(folder_path)

            if not os.path.exists(folder_path):
                return False, f"Path not found: {folder_path}", [], {}

            # Get and filter files
            files = ImageManager.get_image_files(folder_path, include_subfolders, filter_type)
            if not files:
                return False, "No valid images found in the specified path", [], {}
            
            # Sort files
            files = sorted(files, key=numerical_sort_key if sort_method == "numerical" 
                         else os.path.getctime if sort_method == "date_created"
                         else os.path.getmtime if sort_method == "date_modified"
                         else str)

            # Clean up existing thumbnails
            for f in os.listdir(input_dir):
                if f.startswith(ImageManager.THUMBNAIL_PREFIX):
                    try:
                        os.remove(os.path.join(input_dir, f))
                    except Exception as e:
                        logger.warning(f"Error removing old thumbnail {f}: {e}")

            # Apply index and count limits
            start_idx = min(max(0, start_index), len(files))
            end_idx = len(files) if max_images is None else min(start_idx + max_images, len(files))
            selected_files = files[start_idx:end_idx]

            # Create thumbnails with encoded paths - only for selected range
            for idx, file_path in enumerate(selected_files, start=start_idx):
                try:
                    thumb_name = ImageManager.encode_path_to_filename(file_path, folder_path)
                    thumb_path = os.path.join(input_dir, thumb_name)

                    with Image.open(file_path) as img:
                        img = ImageOps.exif_transpose(img)
                        
                        if img.mode in ('RGBA', 'LA'):
                            background = Image.new('RGB', img.size, (255, 255, 255))
                            if img.mode == 'RGBA':
                                background.paste(img, mask=img.split()[3])
                            else:
                                background.paste(img, mask=img.split()[1])
                            img = background
                        elif img.mode not in ('RGB', 'L'):
                            img = img.convert('RGB')

                        img.thumbnail(ImageManager.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                        img.save(thumb_path, "JPEG", quality=70, optimize=True)
                        
                        thumbnail_paths.append(thumb_name)
                        image_order[thumb_name] = idx  # Store image index
                        logger.info(f"Created thumbnail: {thumb_name} for {file_path}")

                except Exception as e:
                    logger.warning(f"Error creating thumbnail for {file_path}: {e}")
                    continue

            if not thumbnail_paths:
                return False, "Failed to create any thumbnails", [], {}

            return True, f"Created {len(thumbnail_paths)} thumbnails", thumbnail_paths, image_order

        except Exception as e:
            logger.error(f"Thumbnail creation failed: {str(e)}")
            return False, f"Thumbnail creation failed: {str(e)}", [], {}
    
    @staticmethod
    def backup_input_folder() -> Tuple[bool, str]:
        try:
            input_dir = folder_paths.get_input_directory()
            backup_dir = os.path.join(os.path.dirname(input_dir), "input_backup")
            
            # Create backup directory if it doesn't exist
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            
            # First, remove all thumbnail files
            for file in os.listdir(input_dir):
                if file.startswith(ImageManager.THUMBNAIL_PREFIX):
                    os.remove(os.path.join(input_dir, file))
            
            # Copy remaining files from input to backup
            for file in os.listdir(input_dir):
                file_path = os.path.join(input_dir, file)
                if os.path.isfile(file_path):
                    shutil.copy2(file_path, backup_dir)
            
            # Clear input directory
            for file in os.listdir(input_dir):
                file_path = os.path.join(input_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    
            return True, "Input folder backed up successfully"
        except Exception as e:
            logger.error(f"Backup failed: {str(e)}")
            return False, f"Backup failed: {str(e)}"

    @staticmethod
    def restore_input_folder() -> Tuple[bool, str]:
        try:
            input_dir = folder_paths.get_input_directory()
            backup_dir = os.path.join(os.path.dirname(input_dir), "input_backup")
            
            if not os.path.exists(backup_dir):
                return False, "Backup directory not found"
            
            # Clear thumbnails first
            for file in os.listdir(input_dir):
                if file.startswith(ImageManager.THUMBNAIL_PREFIX):
                    os.remove(os.path.join(input_dir, file))
            
            # Restore original files
            for file in os.listdir(backup_dir):
                shutil.copy2(os.path.join(backup_dir, file), input_dir)
                
            return True, "Input folder restored successfully"
        except Exception as e:
            logger.error(f"Restore failed: {str(e)}")
            return False, f"Restore failed: {str(e)}"

    @staticmethod
    def sort_files(files: List[str], sort_method: str) -> List[str]:
        """Sort files based on selected method"""
        if sort_method == "numerical":
            return sorted(files, key=numerical_sort_key)
        elif sort_method == "date_created":
            return sorted(files, key=os.path.getctime)
        elif sort_method == "date_modified":
            return sorted(files, key=os.path.getmtime)
        else:  # alphabetical
            return sorted(files)

class IFLoadImagess:
    _color_channels = ["alpha", "red", "green", "blue"]
    
    def __init__(self):
        self.path_cache = {}
        
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        available_images = len([f for f in os.listdir(input_dir) 
                            if f.startswith(ImageManager.THUMBNAIL_PREFIX)])
        available_images = max(1, available_images)
        
        files = [f for f in os.listdir(input_dir) 
                if f.startswith(ImageManager.THUMBNAIL_PREFIX)]
        
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "input_path": ("STRING", {"default": ""}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "stop_index": ("INT", {"default": 10, "min": 1, "max": 9999}),
                "load_limit": (["10", "100", "1000", "10000", "100000"], {"default": "1000"}),
                "image_selected": ("BOOLEAN", {"default": False}),
                "available_image_count": ("INT", {
                    "default": available_images,
                    "min": 0,
                    "max": 99999,
                    "readonly": True
                }),
                "include_subfolders": ("BOOLEAN", {"default": True}),
                "sort_method": (["alphabetical", "numerical", "date_created", "date_modified"],),
                "filter_type": (["none", "png", "jpg", "jpeg", "webp", "gif", "bmp"],),
                "channel": (s._color_channels, {"default": "alpha"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("images", "masks", "image_paths", "filenames", "count_str", "count_int")
    OUTPUT_IS_LIST = (True, True, True, True, True, True)  # Keep as list outputs
    FUNCTION = "load_images"
    CATEGORY = "ImpactFrames💥🎞️"

    @classmethod
    def IS_CHANGED(cls, image, input_path="", start_index=0, stop_index=0, max_images=1,
                include_subfolders=True, sort_method="numerical", image_selected=False, 
                filter_type="none", image_name="", unique_id=None, load_limit="1000", available_image_count=0, channel="alpha"  ):
        """
        Properly handle all input parameters and return NaN to force updates
        This matches the input parameters from INPUT_TYPES
        """
        try:
            # If we have a specific image selected, use its path
            if image and not image.startswith("thb_"):
                image_path = folder_paths.get_annotated_filepath(image)
                if image_path:
                    m = hashlib.sha256()
                    with open(image_path, 'rb') as f:
                        m.update(f.read())
                    return m.digest().hex()
                    
            # For directory-based loads, return NaN to force updates
            return float("NaN")
        except Exception as e:
            logging.warning(f"Error in IS_CHANGED: {e}")
            return float("NaN")

    def load_images(self, image="", input_path="", start_index=0, stop_index=10,
               load_limit="1000", image_selected=False, available_image_count=0, 
               include_subfolders=True, sort_method="numerical", 
               filter_type="none", channel="alpha"):
        try:
            # Process input path
            abs_path = os.path.abspath(input_path if os.path.isabs(input_path) 
                                    else os.path.join(folder_paths.get_input_directory(), input_path))
                    
            # Get all valid images
            all_files = ImageManager.get_image_files(abs_path, include_subfolders, filter_type)
            if not all_files:
                logger.warning(f"No valid images found in {abs_path}")
                img_tensor, mask = self.load_placeholder()
                return ([img_tensor], [mask], [""], [""], ["0/0"], [0])

            # Sort files
            all_files = ImageManager.sort_files(all_files, sort_method)
            total_files = len(all_files)
            
            # Validate indices
            start_index = min(max(0, start_index), total_files)
            stop_index = min(max(start_index + 1, stop_index), total_files)
            num_images = min(stop_index - start_index, int(load_limit))
                
            # Generate thumbnails
            success, _, all_thumbnails, image_order = ImageManager.create_thumbnails(
                abs_path, include_subfolders, filter_type, sort_method,
                start_index=start_index,
                max_images=num_images
            )

            # Handle image selection
            if image_selected and image in image_order:
                start_index = image_order[image]
                num_images = 1

            # Process selected range
            selected_files = all_files[start_index:start_index + num_images]
            
            # Lists to store outputs
            images = []
            masks = []
            paths = []
            filenames = []
            count_strs = []
            count_ints = []

            for idx, file_path in enumerate(selected_files):
                try:
                    img = Image.open(file_path)
                    img = ImageOps.exif_transpose(img)
                    
                    if img.mode == 'I':
                        img = img.point(lambda i: i * (1 / 255))
                    image = img.convert('RGB')
                    
                    # Convert to numpy array and normalize
                    image_array = np.array(image).astype(np.float32) / 255.0
                    # Correct order: [1, H, W, 3]
                    image_tensor = torch.from_numpy(image_array).unsqueeze(0)
                    images.append(image_tensor)
                    
                    # Handle mask based on selected channel
                    if img.mode not in ('RGB', 'L'):
                        img = img.convert('RGBA')
                        
                    c = channel[0].upper()
                    if c in img.getbands():
                        mask = np.array(img.getchannel(c)).astype(np.float32) / 255.0
                        mask = torch.from_numpy(mask)
                        if c == 'A':
                            mask = 1. - mask
                    else:
                        mask = torch.zeros((image_array.shape[0], image_array.shape[1]), 
                                       dtype=torch.float32, device="cpu")
                    
                    masks.append(mask.unsqueeze(0))  # Add batch dimension to mask [1, H, W]
                    
                    paths.append(file_path)
                    filenames.append(os.path.basename(file_path))
                    count_strs.append(f"{start_index + idx + 1}/{total_files}")
                    count_ints.append(start_index + idx + 1)

                except Exception as e:
                    logger.error(f"Error processing image {file_path}: {e}")
                    continue

            if not images:
                img_tensor, mask = self.load_placeholder()
                return ([img_tensor], [mask], [""], [""], ["0/0"], [0])

            return (images, masks, paths, filenames, count_strs, count_ints)

        except Exception as e:
            logger.error(f"Error in load_images: {e}", exc_info=True)
            img_tensor, mask = self.load_placeholder()
            return ([img_tensor], [mask], [""], [""], ["error"], [0])

    def load_placeholder(self):
        """Creates and returns a placeholder image tensor and mask"""
        img = Image.new('RGB', (512, 512), color=(73, 109, 137))
        image_array = np.array(img).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)  # [1, H, W, 3]
        mask = torch.zeros((1, image_array.shape[0], image_array.shape[1]), 
                       dtype=torch.float32, device="cpu")  # [1, H, W]
        return image_tensor, mask

    def process_single_image(self, image_path: str):
        """Process a single image and return appropriate outputs"""
        try:
            img = Image.open(image_path)
            img = ImageOps.exif_transpose(img)
            
            if img.mode == 'I':
                img = img.point(lambda i: i * (1 / 255))
            
            image = img.convert("RGB")
            image_array = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array)[None,]
            
            if 'A' in img.getbands():
                mask = np.array(img.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((image_array.shape[0], image_array.shape[1]), 
                               dtype=torch.float32, device="cpu")
            
            filename = os.path.basename(image_path)
            return ([image_tensor], [mask.unsqueeze(0)], [image_path], [filename], ["1/1"], [1])
            
        except Exception as e:
            logger.error(f"Error processing single image {image_path}: {e}")
            img_tensor, mask = self.load_placeholder()
            return ([img_tensor], [mask], [""], [""], ["error"], [0])
    
@PromptServer.instance.routes.post("/ifai/backup_input")
async def backup_input_folder(request):
    try:
        success, message = ImageManager.backup_input_folder()
        return web.json_response({
            "success": success,
            "message": message
        })
    except Exception as e:
        logger.error(f"Error in backup_input_folder route: {str(e)}")
        return web.json_response({
            "success": False,
            "error": str(e)
        }, status=500)

@PromptServer.instance.routes.post("/ifai/restore_input")
async def restore_input_folder(request):
    try:
        success, message = ImageManager.restore_input_folder()
        return web.json_response({
            "success": success,
            "message": message
        })
    except Exception as e:
        logger.error(f"Error in restore_input_folder route: {str(e)}")
        return web.json_response({
            "success": False,
            "error": str(e)
        }, status=500)

@PromptServer.instance.routes.post("/ifai/refresh_previews")
async def refresh_previews(request):
    try:
        data = await request.json()
        if not data.get("input_path"):
            raise ValueError("No input path provided")

        # Extract parameters
        load_limit = int(data.get("load_limit", "1000"))
        start_index = int(data.get("start_index", 0))
        stop_index = int(data.get("stop_index", 10))
        include_subfolders = data.get("include_subfolders", True)
        filter_type = data.get("filter_type", "none")
        sort_method = data.get("sort_method", "alphabetical")

        # Get files
        if not os.path.isabs(data["input_path"]):
            abs_path = os.path.abspath(os.path.join(folder_paths.get_input_directory(), data["input_path"]))
        else:
            abs_path = data["input_path"]

        # Get all files and sort them
        all_files = ImageManager.get_image_files(abs_path, include_subfolders, filter_type)
        if not all_files:
            return web.json_response({
                "success": False,
                "error": "No valid images found"
            })
        
        all_files = sorted(all_files, 
                         key=numerical_sort_key if sort_method == "numerical" 
                         else os.path.getctime if sort_method == "date_created"
                         else os.path.getmtime if sort_method == "date_modified"
                         else str)
        
        total_available = len(all_files)
        
        # Validate and adjust indices
        start_index = min(max(0, start_index), total_available)
        stop_index = min(max(start_index + 1, stop_index), total_available)
        
        # Calculate how many images to actually load
        num_images = min(stop_index - start_index, load_limit)
            
        # Create thumbnails only for the selected range
        success, message, thumbnails, image_order = ImageManager.create_thumbnails(
            data["input_path"],
            include_subfolders=include_subfolders,
            filter_type=filter_type,
            sort_method=sort_method,
            start_index=start_index,
            max_images=num_images  # Only create thumbnails for the range we want
        )
        
        return web.json_response({
            "success": success,
            "message": message,
            "thumbnails": thumbnails,
            "total_images": total_available,
            "visible_images": len(thumbnails),
            "start_index": start_index,
            "stop_index": stop_index,
            "image_order": image_order
        })

    except Exception as e:
        logger.error(f"Error in refresh_previews route: {str(e)}")
        return web.json_response({
            "success": False,
            "error": str(e)
        }, status=500)

# Add route for widget refresh
@PromptServer.instance.routes.post("/ifai/refresh_widgets")
async def refresh_widgets(request):
    try:
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = sorted(files)
        
        return web.json_response({
            "success": True,
            "files": files
        })
    except Exception as e:
        logger.error(f"Error in refresh_widgets route: {str(e)}")
        return web.json_response({
            "success": False,
            "error": str(e)
        }, status=500)

# Register node class
NODE_CLASS_MAPPINGS = {
    "IF_LoadImagesS": IFLoadImagess
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IF_LoadImagesS": "IF Load Images S 🖼️"
}