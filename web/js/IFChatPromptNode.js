//IFChatPromptNode.js
console.log("IFChatPromptNode extension loading...");

// Add the global error handler
window.addEventListener('error', function(event) {
    console.error('Uncaught error:', event.error);
});
const { app } = window.comfyAPI.app;
//import { app } from "/scripts/app.js";
//import { app } from "/scripts/app.js";

let addRAGWidget;

app.registerExtension({
    name: "Comfy.IFChatPromptNode",

    async init() {
        console.log("IFChatPromptNode init called");
    },

    async setup() {
        try {
            // Wait for UI and API to be ready
            let attempts = 0;
            const maxAttempts = 10;
            const waitTime = 1000; // 1 second
    
            while ((!app.ui?.settings?.store || !app.api) && attempts < maxAttempts) {
                console.log(`Attempt ${attempts + 1}/${maxAttempts}: Waiting for UI and API to initialize...`);
                await new Promise(resolve => setTimeout(resolve, waitTime));
                attempts++;
            }
    
            if (!app.ui?.settings?.store || !app.api) {
                console.warn("UI settings or API not available after maximum attempts");
                // Continue anyway - don't return, as we still want the node to register
            }
    
            // Remove any existing event listeners to prevent duplicates
            if (this.folderInitHandler) {
                app.api.removeEventListener("folder_initialized", this.folderInitHandler);
            }
    
            // Create bound event handler and store reference
            this.folderInitHandler = this.handleFolderInitialized.bind(this);
            app.api?.addEventListener("folder_initialized", this.folderInitHandler);
            console.log("Event listeners set up successfully");
    
        } catch (error) {
            console.error("Error during setup:", error);
            // Don't throw error to allow extension to continue loading
        }
    },   
    
    handleFolderInitialized(event) {
        const data = event.detail;
        console.log("Folder initialized event received:", data);
        if (data.status === "success") {
            console.log(`Folder initialized: ${data.rag_root_dir}`);
            alert(`Folder initialized: ${data.rag_root_dir}`);
            this.updateSelectedNode(data.rag_root_dir);
        } else {
            console.error(`Failed to initialize folder: ${data.message}`);
            alert(`Failed to initialize folder: ${data.message}`);
        }
    },

    updateSelectedNode(rag_root_dir) {
        console.log(`Updating selected node with rag_root_dir: ${rag_root_dir}`);
        const selectedNode = app.graph.getSelectedNodes()[0];
        if (selectedNode && selectedNode.type === "IF_ChatPrompt") {
            selectedNode.properties.rag_root_dir = String(rag_root_dir);
            console.log(`Updated node properties:`, selectedNode.properties);
            selectedNode.setDirtyCanvas(true, true);
        } else {
            console.warn(`No suitable node selected for update`);
        }
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "IF_ChatPrompt") {
            const originalWidget = nodeType.prototype.getWidget;
            nodeType.prototype.getWidget = function (name) {
                let widget = originalWidget.call(this, name);
                if (name === "llm_model" || name === "embedding_model") {
                    widget.combo_type = 1; // Set to multiline dropdown
                }
                return widget;
            };

            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (originalOnNodeCreated) {
                    originalOnNodeCreated.apply(this, arguments);
                }

                LiteGraph = app.graph.constructor;
                const self = this;

                // Define addRAGWidget 
                addRAGWidget = (type, name, options = {}) => {
                    const callback = options.callback ? options.callback.bind(self) : () => {};
                    const widget = self.addWidget(
                        type, 
                        name, 
                        options.default, 
                        callback,
                        {
                            ...options,
                            property: options.property || name.toLowerCase().replace(/\s+/g, '_'),
                            serialize: options.serialize !== undefined ? options.serialize : true
                        }
                    );
                    widget.group = "RAG Settings";
                    return widget;
                };

                

                addRAGWidget("text", "RAG Folder Name", {
                    default: "rag_data",
                    multiline: false,
                    callback: (v) => { this.properties.rag_folder_name = v; }
                });

                // Initialize RAG folder
                addRAGWidget("button", "Initialize RAG Folder", {
                    callback:  () => { 
                        this.initializeRAGFolder(); 
                    },
                    serialize: false
                });
                

                // Upload file for indexing
                addRAGWidget("button", "Upload File for Indexing", {
                    callback: () => {
                        if (!this.properties.rag_root_dir) {
                            alert("Please initialize a RAG folder first.");
                            return;
                        }

                        const input = document.createElement("input");
                        input.type = "file";
                        input.onchange = async (e) => {
                            const file = e.target.files[0];
                            if (file) {
                                const formData = new FormData();
                                formData.append("file", file);
                                formData.append("rag_root_dir", this.properties.rag_folder_name);
                                try {
                                    const response = await fetch("/IF_ChatPrompt/upload_file", {
                                        method: "POST",
                                        body: formData
                                    });
                                    const result = await response.json();
                                    console.log("Upload result:", result);
                                    alert(result.status === "success" ? result.message : "Upload failed: " + result.message);
                                } catch (error) {
                                    console.error("Upload error:", error);
                                    alert("Upload failed: " + error.message);
                                }
                            }
                        };
                        input.click();
                    },
                    serialize: false
                });

                // Run indexing button
                addRAGWidget("button", "Run Indexing", {
                    callback: this.runIndexing.bind(this),
                    serialize: false
                });

                // Load Index button
                addRAGWidget("button", "Load Index", {
                    callback: async function() {
                        if (!this.properties.rag_folder_name) {
                            alert("Please specify a RAG folder name first.");
                            return;
                        }

                        try {
                            const response = await fetch("/IF_ChatPrompt/load_index", {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({
                                    rag_folder_name: this.properties.rag_folder_name,
                                    query_type: this.widgets.find(w => w.name === "query_type").value
                                })
                            });

                            const result = await response.json();
                            
                            if (result.status === "success") {
                                alert(result.message);
                                // Update the node's properties with the loaded index information
                                this.properties.rag_root_dir = result.rag_root_dir;
                                this.setDirtyCanvas(true, true);
                            } else {
                                alert(`Failed to load index: ${result.message}`);
                            }
                        } catch (error) {
                            console.error("Error loading index:", error);
                            alert(`Error loading index: ${error.message}`);
                        }
                    },
                    serialize: false
                });

                // Delete Index button
                addRAGWidget("button", "Delete Index", {
                    callback: async function() {
                        if (!this.properties.rag_root_dir) {
                            alert("Please initialize a RAG folder first.");
                            return;
                        }

                        // Confirm deletion
                        if (!confirm("Are you sure you want to delete this index? This action cannot be undone.")) {
                            return;
                        }

                        try {
                            const response = await fetch("/IF_ChatPrompt/delete_index", {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({
                                    rag_folder_name: this.properties.rag_folder_name
                                })
                            });

                            const result = await response.json();
                            
                            if (result.status === "success") {
                                alert(result.message);
                                this.setDirtyCanvas(true, true);
                            } else {
                                alert(`Failed to delete index: ${result.message}`);
                            }
                        } catch (error) {
                            console.error("Error deleting index:", error);
                            alert(`Error deleting index: ${error.message}`);
                        }
                    },
                    serialize: false
                });

                // Update LLM models dropdown
                const updateLLMModels = async () => {
                    const llmProviderWidget = this.widgets.find((w) => w.name === "llm_provider");
                    const baseIpWidget = this.widgets.find((w) => w.name === "base_ip");
                    const portWidget = this.widgets.find((w) => w.name === "port");
                    const llmModelWidget = this.widgets.find((w) => w.name === "llm_model");
                    const externalLLMApiKeyWidget = this.widgets.find((w) => w.name === "external_api_key");
                
                    if (llmProviderWidget && baseIpWidget && portWidget && llmModelWidget) {
                        const llmProvider = llmProviderWidget.value;
                        const baseIp = baseIpWidget.value;
                        const port = portWidget.value;
                        const externalLLMApiKey = externalLLMApiKeyWidget ? externalLLMApiKeyWidget.value : "";
                
                        const fallbackModels = {
                            openai: ["gpt-40-mini", "gpt-40", "gpt40-0806-loco-vm", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "tts-l-hd", "dall-e-3", "whisper-I"],
                            mistral: ["codestral-mamba-latest", "open-mistral-nemo", "codestral-latest", "mistral-tiny", "mistral-small", "mistral-medium", "mistral-large", "mistral-small-latest", "mistral-medium", "mistral-medium-latest", "mistral-large-latest", "open-mixtral-8x22b", "open-mixtral-8x7b"],
                            groq: ["llava-v1.5-7b-4096-preview", "llama-3.1-8b-instant", "llama3-groq-70b-8192-tool-use-preview", "llama-3.1-70b-versatile", "llama3-groq-8b-8192-tool-use-preview", "distil-whisper-large-v3-en", "whisper-large-v3", "mixtral-8x7b-32768", "gemma-7b-it"],
                            anthropic: ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
                            gemini: ["gemini-1.5-flash", "gemini-1.5-pro-latest"]
                        };
                
                        try {
                            const response = await fetch("/IF_ChatPrompt/get_llm_models", {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ 
                                    llm_provider: llmProvider, 
                                    base_ip: baseIp, 
                                    port,
                                    external_api_key: externalLLMApiKey
                                })
                            });
                
                            let models;
                            if (response.ok) {
                                models = await response.json();
                                if (!Array.isArray(models) || models.length === 0) {
                                    // Wait for 2 seconds before falling back to hardcoded models
                                    await delay(2000);
                                    models = await response.json();
                                    if (!Array.isArray(models) || models.length === 0) {
                                        models = fallbackModels[llmProvider] || ["No models available"];
                                    }
                                }
                            } else {
                                console.error("Failed to fetch LLM models:", await response.text());
                                models = fallbackModels[llmProvider] || ["Error fetching LLM models"];
                            }

                            llmModelWidget.options.values = models;
                            llmModelWidget.value = models[0] || "";
                            app.graph.setDirtyCanvas(true);
                        } catch (error) {
                            console.error("Error fetching LLM models:", error);
                            const models = fallbackModels[llmProvider] || ["Error fetching LLM models"];
                            llmModelWidget.options.values = models;
                            llmModelWidget.value = models[0] || "";
                        }
                    }
                };
                updateLLMModels();
                this.widgets.forEach(w => {
                    if (["llm_provider", "base_ip", "port", "external_api_key"].includes(w.name)) {
                        w.callback = updateLLMModels;
                    }
                });
                
                // Update Embedding models dropdown
                const updateEmbeddingModels = async () => {
                    const embeddingProviderWidget = this.widgets.find((w) => w.name === "embedding_provider");
                    const baseIpWidget = this.widgets.find((w) => w.name === "base_ip");
                    const portWidget = this.widgets.find((w) => w.name === "port");
                    const embeddingModelWidget = this.widgets.find((w) => w.name === "embedding_model");
                    const externalEmbeddingApiKeyWidget = this.widgets.find((w) => w.name === "external_api_key");
                
                    if (embeddingProviderWidget && baseIpWidget && portWidget && embeddingModelWidget) {
                        const embeddingProvider = embeddingProviderWidget.value;
                        const baseIp = baseIpWidget.value;
                        const port = portWidget.value;
                        const externalEmbeddingApiKey = externalEmbeddingApiKeyWidget ? externalEmbeddingApiKeyWidget.value : "";
                
                        const fallbackEmbeddingModels = {
                            openai: ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
                            mistral: ["mistral-embed"],
                            groq: ["no_embedding_models_available"],
                            anthropic: ["no_embedding_models_available"],
                            gemini: ["no_embedding_models_available"]
                        };
                
                        try {
                            const response = await fetch("/IF_ChatPrompt/get_embedding_models", {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ 
                                    embedding_provider: embeddingProvider, 
                                    base_ip: baseIp, 
                                    port,
                                    external_api_key: externalEmbeddingApiKey 
                                })
                            });
                
                            let models;
                            if (response.ok) {
                                models = await response.json();
                                if (!Array.isArray(models) || models.length === 0) {
                                    // Wait for 2 seconds before falling back to hardcoded models
                                    await delay(2000);
                                    models = await response.json();
                                    if (!Array.isArray(models) || models.length === 0) {
                                        models = fallbackEmbeddingModels[embeddingProvider] || fallbackEmbeddingModels.default;
                                    }
                                }
                            } else {
                                console.error("Failed to fetch Embedding models:", await response.text());
                                models = fallbackEmbeddingModels[embeddingProvider] || fallbackEmbeddingModels.default;
                            }
                
                            embeddingModelWidget.options.values = models;
                            embeddingModelWidget.value = models[0] || "";
                            app.graph.setDirtyCanvas(true);
                        } catch (error) {
                            console.error("Error fetching Embedding models:", error);
                            const models = fallbackEmbeddingModels[embeddingProvider] || fallbackEmbeddingModels.default;
                            embeddingModelWidget.options.values = models;
                            embeddingModelWidget.value = models[0] || "";
                        }
                    }
                };
                updateEmbeddingModels();
                this.widgets.forEach(w => {
                    if (["embedding_provider", "base_ip", "port", "external_api_key"].includes(w.name)) {
                        w.callback = updateEmbeddingModels;
                    }
                });
                // Set up onExecuted
                const originalOnExecuted = this.onExecuted;
                this.onExecuted = function (message) {
                    if (originalOnExecuted) {
                        originalOnExecuted.apply(this, arguments);
                    }

                    const data = this.getGraphRAGSettings(); // Get all settings

                    fetch("/IF_ChatPrompt/process_chat", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(data) // Send all settings to the backend
                    })
                    .then(response => response.json())
                    .then(result => {
                        console.log("Chat processing result:", result);
                    })
                    .catch(error => {
                        console.error("Error processing chat:", error);
                    });
                };

                this.setDirtyCanvas(true, true);
            };

            nodeType.prototype.initializeRAGFolder = async function () {
                console.log("Initialize RAG Folder function called");
                const folderName = this.widgets.find(w => w.name === "RAG Folder Name").value.trim();
                if (!folderName) {
                    alert("Please enter a folder name.");
                    return;
                }

                console.log("Initializing RAG folder:", folderName);

                try {
                    const uiSettings = await this.getGraphRAGSettings();
                    console.log("UI Settings:", uiSettings);

                    const response = await fetch("/IF_ChatPrompt/setup_and_initialize", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            folder_name: folderName,
                            ...uiSettings
                        })
                    });

                    if (response.ok) {
                        const result = await response.json();
                        console.log("RAG folder initialization result:", result);

                        if (result.status === "success") {
                            alert(`RAG folder initialized: ${result.rag_root_dir}`);
                            // Update rag_root_dir in node properties
                            this.properties.rag_root_dir = result.rag_root_dir;
                            // Refresh the UI
                            this.setDirtyCanvas(true, true);
                        } else {
                            alert(`Failed to initialize RAG folder: ${result.message}`);
                        }

                    } else {
                        throw new Error(`Failed to initialize folder: ${await response.text()}`);
                    }
                } catch (error) {
                    console.error("Error initializing RAG folder:", error);
                    alert(`Error initializing RAG folder: ${error.message}`);
                }
            };


            nodeType.prototype.runIndexing = function () {
                console.log("Run Indexing function called");
                if (!this.properties.rag_root_dir) {
                    alert("Please initialize a RAG folder first.");
                    return;
                }
                const queryType = this.widgets.find(w => w.name === "query_type").value;

                fetch("/IF_ChatPrompt/run_indexer", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        rag_folder_name: String(this.properties.rag_folder_name),
                        mode_type: String(queryType)
                    })
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    return response.json();
                })
                .then(result => {
                    console.log("Indexing result:", result);
                    alert(result.message);
                })
                .catch(error => {
                    console.error("Error starting indexing:", error);
                    alert("Error starting indexing: " + error.message);
                });
            };


            nodeType.prototype.getGraphRAGSettings = function () {
                const getWidgetValue = (name) => {
                    const widget = this.widgets.find(w => w.name === name);
                    return widget ? widget.value : undefined;
                };

                return {
                    base_ip: getWidgetValue("base_ip"),
                    port: getWidgetValue("port"),
                    llm_provider: getWidgetValue("llm_provider"),
                    llm_model: getWidgetValue("llm_model"),
                    embedding_provider: getWidgetValue("embedding_provider"),
                    embedding_model: getWidgetValue("embedding_model"),
                    temperature: getWidgetValue("temperature"),
                    max_tokens: getWidgetValue("max_tokens"),
                    enable_RAG: getWidgetValue("Enable RAG"),
                    stop: getWidgetValue("Stop"),
                    keep_alive: getWidgetValue("Keep Alive"),
                    top_k: getWidgetValue("Top K"),
                    top_p: getWidgetValue("Top P"),
                    repeat_penalty: getWidgetValue("Repeat Penalty"),
                    rag_folder_name: getWidgetValue("RAG Folder Name"),
                    query_type: getWidgetValue("query_type"),
                    external_api_key: getWidgetValue("External LLM API Key"),
                    seed: getWidgetValue("seed"),
                    prime_directives: getWidgetValue("Prime Directives"),   
                    prompt: getWidgetValue("Prompt"),
                    response_format: getWidgetValue("Response Format"),
                    random: getWidgetValue("Random"),
                    precision: getWidgetValue("Precision"),
                    attention: getWidgetValue("Attention"),
                    aspect_ratio: getWidgetValue("Aspect Ratio"),
                    //selected_folder: this.properties.rag_root_dir,
                    top_k_search: getWidgetValue("Top K Search"),
                };
            };

        }
    }
});

