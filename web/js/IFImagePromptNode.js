//IFImagePromptMkrNode.js
import { app } from "/scripts/app.js";

app.registerExtension({
    name: "Comfy.IFImagePromptNode",
    
    async setup() {
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
        if (nodeData.name === "IF_ImagePrompt") {
            const originalNodeCreated = nodeType.prototype.onNodeCreated;
            
            // Single onNodeCreated implementation that combines all functionality
            nodeType.prototype.onNodeCreated = function() {
                // Call original if it exists
                if (originalNodeCreated) {
                    originalNodeCreated.apply(this, arguments);
                }

                const self = this;

                // Add settings button
                const saveComboSettings = this.addWidget("button", "Store Auto Prompt", null, () => {
                    const settings = this.getNodeComboSettings();
                    
                    fetch("/IF_ImagePrompt/save_combo_settings", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(settings)
                    })
                    .then(response => response.json())
                    .then(result => {
                        if (result.status === "success") {
                            alert("Combo settings saved successfully!");
                        } else {
                            alert("Error saving settings: " + result.message);
                        }
                    })
                    .catch(error => {
                        console.error("Error saving combo settings:", error);
                        alert("Error saving settings: " + error.message);
                    });
                });
                
                // Configure button styling
                saveComboSettings.serialize = false;
                
                // Add LLM model update functionality
                const updateLLMModels = async () => {
                    const llmProviderWidget = this.widgets.find((w) => w.name === "llm_provider");
                    const baseIpWidget = this.widgets.find((w) => w.name === "base_ip");
                    const portWidget = this.widgets.find((w) => w.name === "port");
                    const llmModelWidget = this.widgets.find((w) => w.name === "llm_model");
                    const externalApiKeyWidget = this.widgets.find((w) => w.name === "external_api_key");

                    if (llmProviderWidget && baseIpWidget && portWidget && llmModelWidget) {
                        try {
                            const response = await fetch("/IF_ImagePrompt/get_llm_models", {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({
                                    llm_provider: llmProviderWidget.value,
                                    base_ip: baseIpWidget.value,
                                    port: portWidget.value,
                                    external_api_key: externalApiKeyWidget?.value || ""
                                })
                            });

                            if (!response.ok) {
                                throw new Error(`HTTP error! status: ${response.status}`);
                            }

                            const models = await response.json();
                            console.log("Fetched models:", models);

                            if (Array.isArray(models) && models.length > 0) {
                                llmModelWidget.options.values = models;
                                llmModelWidget.value = models[0];
                                this.setDirtyCanvas(true, true);
                            } else {
                                throw new Error("No models available");
                            }
                        } catch (error) {
                            console.error("Error updating models:", error);
                            
                            // Fallback models
                            const fallbackModels = {
                                openai: ["gpt-4-vision-preview", "gpt-4-1106-vision-preview"],
                                anthropic: ["claude-3-opus-20240229", "claude-3-sonnet-20240229"],
                                ollama: ["llava", "llava-v1.5-7b", "bakllava"]
                            };

                            const models = fallbackModels[llmProviderWidget.value] || ["No models available"];
                            llmModelWidget.options.values = models;
                            llmModelWidget.value = models[0];
                        }
                    }
                };

                // Node settings collection
                this.getNodeComboSettings = function() {
                    const getWidgetValue = (name) => {
                        const widget = this.widgets.find(w => w.name === name);
                        return widget ? widget.value : undefined;
                    };

                    return {
                        llm_provider: getWidgetValue('llm_provider'),
                        llm_model: getWidgetValue('llm_model'),
                        base_ip: getWidgetValue('base_ip'),
                        port: getWidgetValue('port'),
                        user_prompt: getWidgetValue('user_prompt'),
                        prime_directives: getWidgetValue('prime_directives'),
                        temperature: getWidgetValue('temperature'),
                        max_tokens: getWidgetValue('max_tokens'),
                        stop_string: getWidgetValue('stop_string'),
                        keep_alive: getWidgetValue('keep_alive'),
                        top_k: getWidgetValue('top_k'),
                        top_p: getWidgetValue('top_p'),
                        repeat_penalty: getWidgetValue('repeat_penalty'),
                        seed: getWidgetValue('seed'),
                        external_api_key: getWidgetValue('external_api_key'),
                        random: getWidgetValue('random'),
                        precision: getWidgetValue('precision'),
                        attention: getWidgetValue('attention'),
                        aspect_ratio: getWidgetValue('aspect_ratio'),
                        batch_count: getWidgetValue('batch_count'),
                        strategy: getWidgetValue('strategy')
                    };
                };

                // Set up widget callbacks
                this.widgets.forEach(w => {
                    if (["llm_provider", "base_ip", "port", "external_api_key"].includes(w.name)) {
                        w.callback = updateLLMModels;
                    }
                });

                // Initial model update
                updateLLMModels();
            };

            // Add node preview handling
            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function(ctx) {
                if (onDrawForeground) {
                    onDrawForeground.apply(this, arguments);
                }
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

            // Add helper methods
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

            // Handle execution results
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                if (onExecuted) {
                    onExecuted.apply(this, arguments);
                }
                if (message?.generated_prompt) {
                    this.generated_prompt = message.generated_prompt;
                    this.setDirtyCanvas(true, true);
                }
            };
        }
    }
});