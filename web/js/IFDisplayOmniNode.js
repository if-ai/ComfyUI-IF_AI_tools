import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

app.registerExtension({
    name: "Comfy.IFDisplayOmniNode",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "IF_DisplayOmni") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                let nodeName = `${nodeData.name}_${app.graph._nodes.filter(n => n.type == nodeData.name).length}`;

                console.log(`Create ${nodeData.name}: ${nodeName}`);

                const widget = ComfyWidgets.STRING(this, nodeName, ["STRING", {
                    multiline: true,
                    default: "Data will appear here...",
                }], app);
                
                widget.widget.inputEl.readOnly = true;
                widget.widget.inputEl.style.opacity = 0.6;
                
                return ret;
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                if (message?.text) {
                    this.widgets[0].value = message.text;
                    app.graph.setDirtyCanvas(true);
                }
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (config) {
                onConfigure?.apply(this, arguments);
                if (config.widgets_values?.length) {
                    this.widgets[0].value = config.widgets_values[0];
                    app.graph.setDirtyCanvas(true);
                }
            };
        }
    }
});