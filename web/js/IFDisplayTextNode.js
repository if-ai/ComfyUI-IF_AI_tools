import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

app.registerExtension({
    name: "Comfy.IFDisplayTextNode",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "IF_DisplayText") {
            // Add output labels and set shapes
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Set output labels
                if (this.outputs) {
                    // Update labels
                    this.outputs[0].name = "text";
                    this.outputs[1].name = "text_list";
                    this.outputs[2].name = "count";
                    this.outputs[3].name = "selected";
                    
                    // Set the text_list output to use grid shape
                    this.outputs[1].shape = LiteGraph.GRID_SHAPE;
                }

                let IF_DisplayText = app.graph._nodes.filter(wi => wi.type == nodeData.name),
                    nodeName = `${nodeData.name}_${IF_DisplayText.length}`;

                console.log(`Create ${nodeData.name}: ${nodeName}`);

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

            // Add tooltips and visual indicators
            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function (ctx) {
                const ret = onDrawForeground?.apply(this, arguments);
                
                if (this.outputs && this.outputs.length > 0) {
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
                return ret;
            };

            // Function set value
            const outSet = function (texts) {
                if (texts && texts.length > 0) {
                    let widget_id = this?.widgets.findIndex(w => w.type == "customtext");

                    if (Array.isArray(texts)) {
                        texts = texts
                            .filter(word => word != null)
                            .map(word => String(word).trim())
                            .join("\n");
                    } else {
                        texts = String(texts).trim();
                    }

                    this.widgets[widget_id].value = texts;
                    app.graph.setDirtyCanvas(true);
                }
            };

            // onExecuted
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (texts) {
                onExecuted?.apply(this, arguments);
                outSet.call(this, texts?.string);
            };

            // onConfigure
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (w) {
                onConfigure?.apply(this, arguments);
                if (w?.widgets_values?.length) {
                    outSet.call(this, w.widgets_values);
                }
            };
        }
    },
});