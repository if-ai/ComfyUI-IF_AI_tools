import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "IFHFDownload",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType === "IF_HFDownload") {
            nodeData.widgets.download.onClick = async (node) => {
                const mode = node.data.mode;
                const repoId = node.data.repo_id;
                const filePath = node.data.file_path;
                const folderPath = node.data.folder_path;
                const excludeFiles = node.data.exclude_files;
                
                node.setLoading(true);  // Set the node to loading state
                
                try {
                    const res = await app.apiClient.post("/custom_node/hf_download", {
                        mode,
                        repo_id: repoId,
                        file_path: filePath,
                        folder_path: folderPath,
                        exclude_files: excludeFiles
                    });
                    
                    node.data.output = res.data;
                    node.onExecuted?.();
                } catch (error) {
                    console.error("Error downloading from HuggingFace:", error);
                }
                
                node.setLoading(false);  // Set the node back to normal state
            };
        }
    }
});