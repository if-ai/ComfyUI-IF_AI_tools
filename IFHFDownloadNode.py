import os
from huggingface_hub import hf_hub_download, Repository
from server import PromptServer
from aiohttp import web

@PromptServer.instance.routes.post("/custom_node/hf_download")
async def hf_download_handler(request):
    post = await request.post()
    mode = post.get("mode")
    repo_id = post.get("repo_id")
    file_path = post.get("file_path")
    folder_path = post.get("folder_path")
    exclude_files = post.get("exclude_files")

    output = IFHFDownload().download_hf(mode, repo_id, file_path, folder_path, exclude_files)

    return web.json_response(output[0])

class IFHFDownload:
    def __init__(self):
        self.output = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": ("STRING", {"multiline": True}),
                "file_path": ("STRING", {"multiline": True, "condition": "mode == false"}),
                "folder_path": ("STRING", {"multiline": True}),
                "exclude_files": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {                
                "mode": ("BOOLEAN", {"default": False, "label_on": "All Repo", "label_off": "Individual File"}),
            },
            "widgets": {
                "download": ("BUTTON", {"text": "Download"}),
            },
        }
        
    RETURN_TYPES = ("STRING",)
    FUNCTION = "download_hf"
    CATEGORY = "ImpactFrames💥🎞️"

    def download_hf(self, mode, repo_id, file_path, folder_path, exclude_files):
        exclude_list = [f.strip() for f in exclude_files.split(",") if f.strip()]
        
        if mode:
            repo = Repository(local_dir=folder_path, clone_from=repo_id)
            
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file in exclude_list:
                        os.remove(os.path.join(root, file))
            
            self.output = f"Downloaded repo: {repo_id} to {folder_path}"
        else:
            if file_path not in exclude_list:
                filename = hf_hub_download(repo_id=repo_id, filename=file_path, cache_dir=folder_path)
                self.output = f"Downloaded file: {file_path} from {repo_id} to {folder_path}"
            else:
                self.output = f"Skipped file: {file_path} (excluded)"
        
        return (self.output,)


NODE_CLASS_MAPPINGS = {"IF_HFDownload": IFHFDownload}
NODE_DISPLAY_NAME_MAPPINGS = {"IF_HFDownload": "Hugging Face Download🤗"}