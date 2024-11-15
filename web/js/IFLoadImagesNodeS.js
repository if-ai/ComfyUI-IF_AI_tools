// IFLoadImagesNodeS.js
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Create base styles for buttons
const style = document.createElement('style');
style.textContent = `
    .if-button {
        background: var(--comfy-input-bg);
        border: 1px solid var(--border-color);
        color: var(--input-text);
        padding: 4px 12px;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.2s ease;
        margin-right: 5px;
    }
    
    .if-button:hover {
        background: var(--comfy-input-bg-hover);
    }
    
    .if-button:active {
        transform: translateY(1px);
    }
    
    .if-loader {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 30px;
        height: 30px;
        border: 3px solid #666;
        border-top: 3px solid #fff;
        border-radius: 50%;
        animation: ifSpin 1s linear infinite;
        display: none;
        z-index: 1000;
    }
    
    @keyframes ifSpin {
        0% { transform: translate(-50%, -50%) rotate(0deg); }
        100% { transform: translate(-50%, -50%) rotate(360deg); }
    }
`;
document.head.appendChild(style);


function updateNodePreview(node, imageName) {
    if (!imageName || !node) return;
    
    const img = new Image();
    img.onload = () => {
        node.imgs = [img];
        app.graph.setDirtyCanvas(true);
    };

    img.onerror = () => {
        console.warn(`Failed to load preview for ${imageName}`);
    };

    // Get the input directory path
    const inputPathWidget = node.widgets?.find(w => w.name === "input_path");
    const inputPath = inputPathWidget?.value || "";

    // Construct URL with proper path handling
    const params = `&type=input${app.getPreviewFormatParam?.() || ""}${app.getRandParam?.() || ""}`;
    
    // Use the thumbnail name directly since it's already in the input directory
    img.src = api.apiURL(`/view?filename=${encodeURIComponent(imageName)}&type=input${params}`);
}

app.registerExtension({
    name: "Comfy.IFLoadImagesNodeS",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "IF_LoadImagesS") return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function() {
            const result = origOnNodeCreated?.apply(this, arguments);

            // Image widget setup
            const imageWidget = this.widgets?.find(w => w.name === "image");
            if (imageWidget) {
                // Store original methods
                const origSetValue = imageWidget.setValue;
                const origCallback = imageWidget.callback;

                imageWidget.setValue = function(v, skip_callback) {
                    const result = origSetValue?.call(this, v, skip_callback);
                    if (v && !skip_callback) {
                        updateNodePreview(this.node, v);
                    }
                    return result;
                };

                // Update preview on value change
                imageWidget.callback = function(value) {
                    if (origCallback) {
                        origCallback.call(this, value);
                    }
                    updateNodePreview(this.node, this.value);
                };
            }
        
            // Add management buttons
            const selectFolderBtn = this.addWidget("button", "select_folder", "Select Folder 📂", () => {
                const input = document.createElement("input");
                input.type = "file";
                input.webkitdirectory = true;
                input.directory = true;
                input.style.display = "none";
                document.body.appendChild(input);
        
                input.onchange = async (e) => {
                    try {
                        if (!e.target.files.length) return;
                        const files = e.target.files;
                        
                        // Get folder path
                        let folderPath = '';
                        if (files[0].path) {
                            folderPath = files[0].path.split(/(\/|\\)/);
                            folderPath = folderPath.slice(0, -2).join('');
                        } else if (files[0].webkitRelativePath) {
                            folderPath = files[0].webkitRelativePath.split('/')[0];
                        }
        
                        if (!folderPath) {
                            throw new Error("Could not determine folder path");
                        }
        
                        // Prompt user to confirm/edit path
                        const userFullPath = prompt("Confirm or edit the full path:", folderPath);
                        if (!userFullPath) {
                            throw new Error("No path entered");
                        }
        
                        // Update input_path widget
                        const pathWidget = this.widgets.find(w => w.name === "input_path");
                        if (pathWidget) {
                            pathWidget.value = userFullPath;
                            if (pathWidget.callback) pathWidget.callback(userFullPath);
                            
                            if (pathWidget.inputEl) {
                                pathWidget.inputEl.value = userFullPath;
                            }
                        }
        
                        // Auto-refresh previews
                        const refreshBtn = this.widgets.find(w => w.name === "refresh_preview");
                        if (refreshBtn?.callback) {
                            setTimeout(() => refreshBtn.callback(), 100);
                        }
        
                    } catch (error) {
                        console.error("Folder selection error:", error);
                        alert(error.message);
                    } finally {
                        document.body.removeChild(input);
                    }
                };
        
                input.click();
            });
        
        
            const backupBtn = this.addWidget("button", "backup_input", "Backup Input 💾", 
                async () => {
                    try {
                        const response = await api.fetchApi("/ifai/backup_input", {
                            method: "POST"
                        });
                        
                        if (!response.ok) throw new Error(await response.text());
                        const result = await response.json();
                        
                        if (!result.success) {
                            throw new Error(result.error);
                        }
                        
                        alert("Input folder backed up successfully");
                    } catch (error) {
                        console.error("Backup error:", error);
                        alert(error.message);
                    }
                }
            );
        
            const restoreBtn = this.addWidget("button", "restore_input", "Restore Input ♻️", 
                async () => {
                    try {
                        const response = await api.fetchApi("/ifai/restore_input", {
                            method: "POST"
                        });
                        
                        if (!response.ok) throw new Error(await response.text());
                        const result = await response.json();
                        
                        if (!result.success) {
                            throw new Error(result.error);
                        }
                        
                        alert("Input folder restored successfully");
                        
                        // Refresh previews after restore
                        const refreshBtn = this.widgets.find(w => w.name === "refresh_preview");
                        if (refreshBtn?.callback) {
                            setTimeout(() => refreshBtn.callback(), 100);
                        }
                    } catch (error) {
                        console.error("Restore error:", error);
                        alert(error.message);
                    }
                }
            );

            // In the refresh button callback
            const refreshBtn = this.addWidget("button", "refresh_preview", "Refresh Previews 🔄", async () => {
                try {
                    const inputPath = this.widgets.find(w => w.name === "input_path")?.value;
                    if (!inputPath) {
                        alert("Please select a folder first");
                        return;
                    }
            
                    // Get widget values
                    const startIndexWidget = this.widgets.find(w => w.name === "start_index");
                    const stopIndexWidget = this.widgets.find(w => w.name === "stop_index");
                    
                    if (stopIndexWidget.value <= startIndexWidget.value) {
                        alert("Stop index must be greater than start index");
                        return;
                    }
            
                    const options = {
                        input_path: inputPath,
                        include_subfolders: this.widgets.find(w => w.name === "include_subfolders")?.value ?? true,
                        sort_method: this.widgets.find(w => w.name === "sort_method")?.value ?? "alphabetical",
                        filter_type: this.widgets.find(w => w.name === "filter_type")?.value ?? "none",
                        start_index: startIndexWidget.value,
                        stop_index: stopIndexWidget.value,
                        load_limit: parseInt(this.widgets.find(w => w.name === "load_limit")?.value || "1000")
                    };
            
                    const response = await api.fetchApi("/ifai/refresh_previews", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(options)
                    });
            
                    if (!response.ok) throw new Error(await response.text());
                    const result = await response.json();
                    
                    if (!result.success) throw new Error(result.error);
            
                    // Update widgets
                    const imageWidget = this.widgets?.find(w => w.name === "image");
                    const availableCountWidget = this.widgets?.find(w => w.name === "available_image_count");
                    
                    if (imageWidget && result.thumbnails?.length) {
                        imageWidget.options.values = result.thumbnails;
                        imageWidget.value = result.thumbnails[0];
                        this.imageOrder = result.image_order || {};
                        imageWidget.callback.call(imageWidget);
                    }
                    
                    // Update the available count
                    if (availableCountWidget) {
                        availableCountWidget.value = result.total_images;
                    }
            
                    // Update stop_index if needed
                    if (result.stop_index !== stopIndexWidget.value) {
                        stopIndexWidget.value = result.stop_index;
                    }
            
                } catch (error) {
                    console.error("Error refreshing previews:", error);
                    alert("Error refreshing previews: " + error.message);
                }
            });

            
            // Arrange widgets - move refresh button below image widget
            if (refreshBtn) {
                const widgets = this.widgets.splice(-1); // Only remove one widget (refreshBtn)
                this.widgets.splice(imageWidget ? this.widgets.indexOf(imageWidget) + 1 : 0, 0, ...widgets);
            }

        
            // Handle execution results
            this.onExecuted = function(output) {
                if (output?.ui?.values) {
                    const imageWidget = this.widgets?.find(w => w.name === "image");
                    const availableCountWidget = this.widgets?.find(w => w.name === "available_image_count");
                    const maxImagesWidget = this.widgets?.find(w => w.name === "max_images");
                    
                    if (imageWidget) {
                        // Store path mapping and image order
                        this.pathMapping = output.ui.values.path_mapping || {};
                        this.imageOrder = output.ui.values.image_order || {};
                        
                        // Update widget options
                        if (output.ui.values.images) {
                            imageWidget.options.values = output.ui.values.images;
                            
                            // Update available count and limits
                            const count = output.ui.values.available_image_count;
                            if (availableCountWidget) {
                                availableCountWidget.value = count;
                            }
                            if (maxImagesWidget) {
                                maxImagesWidget.options.max = count;
                                if (maxImagesWidget.value > count) {
                                    maxImagesWidget.value = count;
                                }
                            }
                        }
                        
                        // Handle current selection
                        if (output.ui.values.current_thumbnails?.length > 0) {
                            const currentValue = imageWidget.value;
                            if (!this.pathMapping[currentValue]) {
                                imageWidget.value = output.ui.values.current_thumbnails[0];
                            }
                        }
                        
                        // Update preview
                        if (imageWidget.value) {
                            updateNodePreview(this, imageWidget.value);
                        }
                    }
                }
            };
        
            return result;
        };
        // API calls
        nodeType.prototype.backupInputFolder = async function() {
            try {
                this.showLoader();
                const response = await fetch("/ifai/backup_input", {
                    method: "POST"
                });
                
                if (!response.ok) throw new Error(await response.text());
                const result = await response.json();
                
                if (!result.success) {
                    throw new Error(result.error);
                }
                
                this.showMessage("Input folder backed up successfully");
            } catch (error) {
                console.error("Backup error:", error);
                this.showMessage(error.message, "error");
            } finally {
                this.hideLoader();
            }
        };

        nodeType.prototype.restoreInputFolder = async function() {
            try {
                this.showLoader();
                const response = await fetch("/ifai/restore_input", {
                    method: "POST"
                });
                
                if (!response.ok) throw new Error(await response.text());
                const result = await response.json();
                
                if (!result.success) {
                    throw new Error(result.error);
                }
                
                this.showMessage("Input folder restored successfully");
            } catch (error) {
                console.error("Restore error:", error);
                this.showMessage(error.message, "error");
            } finally {
                this.hideLoader();
            }
        };

        nodeType.prototype.refreshPreviews = async function() {
            try {
                const input_path = this.widgets.find(w => w.name === "input_path")?.value;
                if (!input_path) {
                    throw new Error("Please select a folder first");
                }
                
                const options = {};
                this.widgets.forEach(w => {
                    if (["select_folder", "refresh_preview", "backup_input", "restore_input"].includes(w.name)) return;
                    options[w.name] = w.value;
                });

                this.showLoader();

                const response = await fetch("/ifai/refresh_previews", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(options)
                });

                if (!response.ok) throw new Error(await response.text());
                const result = await response.json();
                
                if (!result.success) {
                    throw new Error(result.error);
                }

                this.showMessage(`Generated ${result.thumbnails.length} previews`);
            } catch (error) {
                console.error("Preview refresh error:", error);
                this.showMessage(error.message, "error");
            } finally {
                this.hideLoader();
            }
        };

        // Handle widget changes
        const origOnWidgetChanged = nodeType.prototype.onWidgetChanged;
        nodeType.prototype.onWidgetChanged = function (name, value) {
            if (origOnWidgetChanged) {
                origOnWidgetChanged.apply(this, arguments);
            }

            // Auto-refresh on certain changes
            if (["include_subfolders", "filter_type", "sort_method"].includes(name)) {
                this.refreshPreviews();
            }
        };

        // Add right-click menu options
        const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            if (origGetExtraMenuOptions) {
                origGetExtraMenuOptions.apply(this, arguments);
            }

            options.unshift(
                {
                    content: "Select Folder",
                    callback: () => {
                        const btn = this.widgets.find(w => w.name === "select_folder");
                        if (btn?.callback) btn.callback();
                    }
                },
                {
                    content: "Refresh Previews",
                    callback: () => {
                        const btn = this.widgets.find(w => w.name === "refresh_preview");
                        if (btn?.callback) btn.callback();
                    }
                },
                {
                    content: "Backup Input Folder",
                    callback: () => {
                        const btn = this.widgets.find(w => w.name === "backup_input");
                        if (btn?.callback) btn.callback();
                    }
                },
                {
                    content: "Restore Input Folder",
                    callback: () => {
                        const btn = this.widgets.find(w => w.name === "restore_input");
                        if (btn?.callback) btn.callback();
                    }
                },
                null // separator
            );

            return options;
        };
    }
});
