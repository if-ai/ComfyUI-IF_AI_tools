//IFPromptMkrNode.js
import { app } from "/scripts/app.js";

app.registerExtension({
    name: "Comfy.IFPromptMkrNode",
    
    async setup() {
        // Wait for UI and API to be ready
        let attempts = 0;
        const maxAttempts = 10;
        const waitTime = 1000;

        while ((!app.ui?.settings?.store || !app.api) && attempts < maxAttempts) {
            console.log(`Attempt ${attempts + 1}/${maxAttempts}: Waiting for UI and API to initialize...`);
            await new Promise(resolve => setTimeout(resolve, waitTime));
            attempts++;
        }
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "IF_PromptMkr") {
            const originalNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                if (originalNodeCreated) {
                    originalNodeCreated.apply(this, arguments);
                }

                const self = this;

                const updateLLMModels = async () => {
                    const llmProviderWidget = this.widgets.find((w) => w.name === "llm_provider");
                    const baseIpWidget = this.widgets.find((w) => w.name === "base_ip");
                    const portWidget = this.widgets.find((w) => w.name === "port");
                    const llmModelWidget = this.widgets.find((w) => w.name === "llm_model");
                    const externalApiKeyWidget = this.widgets.find((w) => w.name === "external_api_key");

                    if (llmProviderWidget && baseIpWidget && portWidget && llmModelWidget) {
                        const llmProvider = llmProviderWidget.value;
                        const baseIp = baseIpWidget.value;
                        const port = portWidget.value;
                        const externalApiKey = externalApiKeyWidget ? externalApiKeyWidget.value : "";

                        try {
                            const response = await fetch("/IF_PromptMkr/get_llm_models", {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({
                                    llm_provider: llmProvider,
                                    base_ip: baseIp,
                                    port: port,
                                    external_api_key: externalApiKey
                                })
                            });

                            if (!response.ok) {
                                throw new Error(`HTTP error! status: ${response.status}`);
                            }

                            const models = await response.json();
                            console.log("Fetched models:", models);

                            if (!Array.isArray(models) || models.length === 0) {
                                throw new Error("No models available");
                            }

                            llmModelWidget.options.values = models;
                            llmModelWidget.value = models[0];
                            
                            // Force widget update
                            this.setDirtyCanvas(true, true);

                        } catch (error) {
                            console.error("Error updating models:", error);
                            
                            // Fallback models based on provider
                            const fallbackModels = {
                                openai: ["gpt-4-vision-preview", "gpt-4-1106-vision-preview"],
                                anthropic: ["claude-3-opus-20240229", "claude-3-sonnet-20240229"],
                                ollama: ["llava", "llava-v1.5-7b", "bakllava"]
                            };

                            const models = fallbackModels[llmProvider] || ["No models available"];
                            llmModelWidget.options.values = models;
                            llmModelWidget.value = models[0];
                        }
                    }
                };

                // Set up widget callbacks
                this.widgets.forEach(w => {
                    if (["llm_provider", "base_ip", "port", "external_api_key"].includes(w.name)) {
                        w.callback = updateLLMModels;
                    }
                });

                // Add custom contextMenu options
                const getExtraMenuOptions = this.getExtraMenuOptions;
                this.getExtraMenuOptions = function() {
                    const options = [];  // Initialize empty array even if original returns nothing
                    
                    // Call original if it exists
                    if (getExtraMenuOptions) {
                        const originalOptions = getExtraMenuOptions.apply(this, arguments);
                        if (Array.isArray(originalOptions)) {
                            options.push(...originalOptions);
                        }
                    }
                    
                    // Add our custom options
                    options.push(
                        {
                            content: "🔄 Refresh Models",
                            callback: () => {
                                updateLLMModels();
                            }
                        },
                        null // divider
                    );

                    // Add copy options only if those widgets exist
                    const systemPromptWidget = this.widgets.find(w => w.name === "system_prompt");
                    if (systemPromptWidget) {
                        options.push({
                            content: "📋 Copy System Prompt",
                            callback: () => {
                                navigator.clipboard.writeText(systemPromptWidget.value);
                            }
                        });
                    }

                    const userPromptWidget = this.widgets.find(w => w.name === "user_prompt");
                    if (userPromptWidget) {
                        options.push(
                            null, // divider
                            {
                                content: "📋 Copy User Prompt",
                                callback: () => {
                                    navigator.clipboard.writeText(userPromptWidget.value);
                                }
                            }
                        );
                    }
                    
                    return options;
                };

                // Initial model update
                updateLLMModels();
            };

            // Add routes to the server
            try {
                const addPromptMkrRoutes = async () => {
                    const response = await fetch("/IF_PromptMkr/add_routes", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({})
                    });
                    if (!response.ok) {
                        console.warn("Failed to add PromptMkr routes");
                    }
                };
                addPromptMkrRoutes();
            } catch (error) {
                console.warn("Error adding PromptMkr routes:", error);
            }

            // Add node preview handling
            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function(ctx) {
                if (onDrawForeground) {
                    onDrawForeground.apply(this, arguments);
                }
                
                // Draw preview of the last generated prompt if available
                if (this.generated_prompt) {
                    const margin = 10;
                    const textX = this.pos[0] + margin;
                    const textY = this.pos[1] + this.size[1] + 20;
                    const maxWidth = this.size[0] - margin * 2;
                    
                    ctx.save();
                    ctx.font = "12px Arial";
                    ctx.fillStyle = "#CCC";
                    this.wrapText(ctx, this.generated_prompt, textX, textY, maxWidth, 16);
                    ctx.restore();
                }
            };

            // Helper method for text wrapping
            nodeType.prototype.wrapText = function(ctx, text, x, y, maxWidth, lineHeight) {
                const words = text.split(' ');
                let line = '';
                let posY = y;

                for (const word of words) {
                    const testLine = line + word + ' ';
                    const metrics = ctx.measureText(testLine);
                    const testWidth = metrics.width;

                    if (testWidth > maxWidth && line !== '') {
                        ctx.fillText(line, x, posY);
                        line = word + ' ';
                        posY += lineHeight;
                    } else {
                        line = testLine;
                    }
                }
                ctx.fillText(line, x, posY);
            };

            // Handle node execution
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                if (onExecuted) {
                    onExecuted.apply(this, arguments);
                }

                // Store generated prompt for preview
                if (message?.generated_prompt) {
                    this.generated_prompt = message.generated_prompt;
                    this.setDirtyCanvas(true, true);
                }
            };
        }
    }
});
