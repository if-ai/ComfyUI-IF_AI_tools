import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

app.registerExtension({
    name: "Comfy.IFStepCounter",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "IF_StepCounter") {
            // Add output labels and tooltips
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const ret = onNodeCreated?.apply(this, arguments);
                
                if (this.outputs) {
                    const outputTooltips = [
                        "Number output (integer or float based on type)",
                        "Float representation of the counter",
                        "Integer representation of the counter",
                        "String representation of the counter"
                    ];
                    
                    // Set tooltips and colors for outputs
                    for (let i = 0; i < this.outputs.length; i++) {
                        const output = this.outputs[i];
                        output.tooltip = outputTooltips[i];
                    }
                }
                
                return ret;
            };

            // Add mode descriptions
            const modes = {
                "increment": "Increases by step each time",
                "decrement": "Decreases by step each time",
                "increment_to_stop": "Increases until reaching stop value",
                "decrement_to_stop": "Decreases until reaching stop value"
            };

            // Add tooltips for inputs
            const orig_getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function(canvas, options) {
                if (orig_getExtraMenuOptions) {
                    orig_getExtraMenuOptions.apply(this, arguments);
                }

                options.push({
                    content: "Mode Descriptions",
                    has_submenu: true,
                    callback: () => {},
                    submenu: {
                        options: Object.entries(modes).map(([mode, desc]) => ({
                            content: `${mode}: ${desc}`,
                            callback: () => {}
                        }))
                    }
                });
            };
        }
    }
});