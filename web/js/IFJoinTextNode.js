import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

app.registerExtension({
    name: "Comfy.IF_JoinTextNode",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "IF_JoinText") {
            // Add output labels and tooltips
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const ret = onNodeCreated?.apply(this, arguments);
                
                if (this.outputs) {
                    this.outputs[0].name = "joined_text";
                    this.outputs[0].tooltip = "Combined text output with separator";
                }
                
                // Add tooltips to inputs
                if (this.inputs) {
                    const inputTooltips = {
                        "separator": "Text to insert between joined strings (e.g., space, comma, newline)",
                        "text1": "First text to join",
                        "text2": "Second text to join",
                        "text3": "Third text to join",
                        "text4": "Fourth text to join"
                    };
                    
                    for (const input of this.inputs) {
                        input.tooltip = inputTooltips[input.name] || input.name;
                    }
                }
                
                return ret;
            };

            // Add helper widget to show current join preview
            const widget = {
                name: "preview",
                type: "text",
                defaultValue: "",
                options: { multiline: true, readonly: true },
            };

            // Update preview when inputs change
            const originalOnExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(output) {
                originalOnExecuted?.apply(this, arguments);
                
                if (this.widgets) {
                    const previewWidget = this.widgets.find(w => w.name === "preview");
                    if (previewWidget && output) {
                        previewWidget.value = `Joined Text Preview:\n${output}`;
                    }
                }
            };

            // Add examples to context menu
            const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function(_, options) {
                getExtraMenuOptions?.apply(this, arguments);
                
                options.push(
                    {
                        content: "Separator Examples",
                        has_submenu: true,
                        callback: () => {},
                        submenu: {
                            options: [
                                { content: "Space: ' '", callback: () => this.widgets[0].value = " " },
                                { content: "Comma: ','", callback: () => this.widgets[0].value = "," },
                                { content: "Newline: '\\n'", callback: () => this.widgets[0].value = "\n" },
                                { content: "Nothing: ''", callback: () => this.widgets[0].value = "" },
                            ]
                        }
                    }
                );
            };
        }
    }
});