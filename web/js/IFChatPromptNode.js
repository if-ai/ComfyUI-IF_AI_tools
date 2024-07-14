import { app } from "/scripts/app.js";

app.registerExtension({
    name: "Comfy.IFChatPromptNode",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "IF_ChatPrompt") {
            const originalWidget = nodeType.prototype.getWidget;
            nodeType.prototype.getWidget = function (name) {
                let widget = originalWidget.call(this, name);
                if (name === "model") {
                    widget.combo_type = 1; // Set to multiline dropdown
                }
                return widget;
            };

            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (originalOnNodeCreated) {
                    originalOnNodeCreated.apply(this, arguments);
                }

                const updateModels = async () => {
                  const engineWidget = this.widgets.find((w) => w.name === "engine");
                  const baseIpWidget = this.widgets.find((w) => w.name === "base_ip");
                  const portWidget = this.widgets.find((w) => w.name === "port");
                  const modelWidget = this.widgets.find((w) => w.name === "model");
              
                  if (engineWidget && baseIpWidget && portWidget && modelWidget) {
                      const engine = engineWidget.value;
                      const baseIp = baseIpWidget.value;
                      const port = portWidget.value;
              
                      try {
                          const response = await fetch("/IF_ChatPrompt/get_models", {
                              method: "POST",
                              headers: { "Content-Type": "application/json" },
                              body: JSON.stringify({ engine, base_ip: baseIp, port })
                          });
              
                          if (response.ok) {
                              const models = await response.json();
                              if (Array.isArray(models) && models.length > 0) {
                                  modelWidget.options.values = models;
                                  modelWidget.value = models[0] || "";
                              } else {
                                  modelWidget.options.values = ["No models available"];
                                  modelWidget.value = "No models available";
                              }
                              app.graph.setDirtyCanvas(true);
                          } else {
                              console.error("Failed to fetch models:", await response.text());
                              modelWidget.options.values = ["Error fetching models"];
                              modelWidget.value = "Error fetching models";
                          }
                      } catch (error) {
                          console.error("Error fetching models:", error);
                          modelWidget.options.values = ["Error fetching models"];
                          modelWidget.value = "Error fetching models";
                      }
                  }
              };

                this.widgets.forEach(w => {
                    if (["engine", "base_ip", "port"].includes(w.name)) {
                        w.callback = updateModels;
                    }
                });

                // Initial update
                updateModels();
            };
        }
    }
});
