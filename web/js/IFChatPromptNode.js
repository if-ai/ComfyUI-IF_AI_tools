import { app } from "/scripts/app.js";

app.registerExtension({
    name: "Comfy.IFChatPromptNode",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "IF_ChatPrompt") {
      const originalNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = async function () {
        if (originalNodeCreated) {
          originalNodeCreated.apply(this, arguments);
        }

        const engineWidget = this.widgets.find((w) => w.name === "engine");
        const modelWidget = this.widgets.find((w) => w.name === "selected_model");
        const baseIpWidget = this.widgets.find((w) => w.name === "base_ip");
        const portWidget = this.widgets.find((w) => w.name === "port");

        const fetchModels = async (engine, baseIp, port) => {
          try {
            const response = await fetch("/IF_ChatPrompt/get_models", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                engine: engine,
                base_ip: baseIp,
                port: port,
              }),
            });

            if (response.ok) {
              const models = await response.json();
              console.log("Fetched models:", models);
              return models;
            } else {
              console.error(`Failed to fetch models: ${response.status}`);
              return [];
            }
          } catch (error) {
            console.error(`Error fetching models for engine ${engine}:`, error);
            return [];
          }
        };

        const updateModels = async () => {
          const engine = engineWidget.value;
          const baseIp = baseIpWidget.value;
          const port = portWidget.value;

          console.log(`Selected engine: ${engine}`);
          console.log(`Base IP: ${baseIp}`);
          console.log(`Port: ${port}`);

          const models = await fetchModels(engine, baseIp, port);

          // Update modelWidget options and value
          modelWidget.options.values = models;
          console.log("Updated modelWidget.options.values:", modelWidget.options.values);

          if (models.includes(modelWidget.value)) {
            modelWidget.value = modelWidget.value;
          } else if (models.length > 0) {
            modelWidget.value = models[0];
          } else {
            modelWidget.value = "";
          }
          console.log("Updated modelWidget.value:", modelWidget.value);

          this.triggerSlot(0);
        };

        engineWidget.callback = updateModels;

        // Initial update
        await updateModels();
      };
    }
  },
});
