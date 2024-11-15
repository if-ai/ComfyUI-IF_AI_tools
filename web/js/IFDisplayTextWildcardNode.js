import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

app.registerExtension({
    name: "Comfy.IFDisplayTextWildcardNode",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "IF_DisplayTextWildcard") {
            // Store original node methods
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            const onExecuted = nodeType.prototype.onExecuted;
            const onDrawForeground = nodeType.prototype.onDrawForeground;
            const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;

            // Add counter status display
            nodeType.prototype.counterStatus = {
                count: -1,
                blocked: false
            };

            nodeType.prototype.onNodeCreated = function () {
                const ret = onNodeCreated?.apply(this, arguments);

                // Set output labels
                if (this.outputs) {
                    this.outputs[0].name = "text";
                    this.outputs[1].name = "text_list";
                    this.outputs[2].name = "count";
                    this.outputs[3].name = "selected";
                    
                    // Set the text_list output to use grid shape
                    this.outputs[1].shape = LiteGraph.GRID_SHAPE;
                }

                // Create unique name for node instance
                let nodeName = `${nodeData.name}_${app.graph._nodes.filter(n => n.type === nodeData.name).length}`;
                console.log(`Create ${nodeData.name}: ${nodeName}`);

                // Create text display widget
                const wi = ComfyWidgets.STRING(
                    this,
                    nodeName,
                    ["STRING", {
                        default: "",
                        placeholder: "Message will appear here ...",
                        multiline: true,
                    }],
                    app
                );
                wi.widget.inputEl.readOnly = true;

                return ret;
            };

            // Handle counter status updates
            nodeType.prototype.onExecuted = function (message) {
                const ret = onExecuted?.apply(this, arguments);
                
                // Update counter status
                const widgets = this.widgets || [];
                const counterWidget = widgets.find(w => w.name === "counter");
                
                if (counterWidget) {
                    // Get counter value from widget
                    this.counterStatus.count = counterWidget.value;
                    
                    // Check if execution was blocked
                    if (message?.error?.includes("Counter reached 0")) {
                        this.counterStatus.blocked = true;
                    } else {
                        this.counterStatus.blocked = false;
                    }
                }

                // Update text display
                if (message?.string) {
                    const widget_id = this.widgets.findIndex(w => w.type === "customtext");
                    if (widget_id !== -1) {
                        let texts = message.string;
                        if (Array.isArray(texts)) {
                            texts = texts
                                .filter(text => text != null)
                                .map(text => String(text).trim())
                                .join("\n");
                        } else {
                            texts = String(texts).trim();
                        }
                        this.widgets[widget_id].value = texts;
                    }
                }

                app.graph.setDirtyCanvas(true);
                return ret;
            };

            // Enhanced node drawing
            nodeType.prototype.onDrawForeground = function (ctx) {
                const ret = onDrawForeground?.apply(this, arguments);

                // Add tooltips to outputs
                if (this.outputs) {
                    const outputLabels = [
                        "Complete Text",
                        "List of Lines",
                        "Line Count",
                        "Selected Line"
                    ];
                    const outputTooltips = [
                        "Full text content",
                        "Individual lines as separate outputs",
                        "Total number of non-empty lines",
                        "Currently selected line based on select input"
                    ];
                    
                    for (let i = 0; i < this.outputs.length; i++) {
                        const output = this.outputs[i];
                        output.tooltip = outputTooltips[i];
                    }
                }

                // Draw counter status
                if (this.counterStatus.count !== -1) {
                    ctx.save();
                    ctx.font = "12px Arial";
                    ctx.textAlign = "right";
                    
                    // Position in top-right corner
                    const text = `Counter: ${this.counterStatus.count}`;
                    const x = this.size[0] - 5;
                    const y = 20;
                    
                    // Draw counter value with color based on status
                    if (this.counterStatus.blocked) {
                        ctx.fillStyle = "#ff4444";
                        this.boxcolor = "#442222";
                    } else if (this.counterStatus.count === 0) {
                        ctx.fillStyle = "#ffaa44";
                        this.boxcolor = "#443322";
                    } else {
                        ctx.fillStyle = "#66ff66";
                        this.boxcolor = null;  // Reset to default
                    }
                    
                    ctx.fillText(text, x, y);
                    ctx.restore();
                }

                return ret;
            };

            // Add counter reset option to context menu
            nodeType.prototype.getExtraMenuOptions = function (_, options) {
                const ret = getExtraMenuOptions?.apply(this, arguments);
                
                options.push(
                    null, // separator
                    {
                        content: "Reset Counter",
                        callback: () => {
                            const counterWidget = this.widgets.find(w => w.name === "counter");
                            if (counterWidget) {
                                // Reset to widget's original value
                                counterWidget.value = counterWidget.options.default;
                                this.counterStatus.count = counterWidget.value;
                                this.counterStatus.blocked = false;
                                app.graph.setDirtyCanvas(true);
                            }
                        }
                    }
                );

                return ret;
            };
        }
    },
});