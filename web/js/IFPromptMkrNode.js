import { app } from "/scripts/app.js";

app.registerExtension({
  name: "Comfy.IFPromptMkrNode",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "IF_PromptMkr") {
      const originalNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = async function () {
        if (originalNodeCreated) {
          originalNodeCreated.apply(this, arguments);
        }
    
        const engineWidget = this.widgets.find((w) => w.name === "engine");
        const modelWidget = this.widgets.find((w) => w.name === "selected_model"); 
        const baseIpWidget = this.widgets.find((w) => w.name === "base_ip");
        const ollamaPortWidget = this.widgets.find((w) => w.name === "ollama_port");
    
        const updateModels = async () => {
          const engine = engineWidget.value;
          const baseIp = baseIpWidget.value;
          const ollamaPort = ollamaPortWidget.value;
        
          console.log(`Selected engine: ${engine}`);
          console.log(`Base IP: ${baseIp}`);
          console.log(`Ollama Port: ${ollamaPort}`);
        
          try {
            let models = [];
        
            if (engine === "ollama") {
              const response = await fetch(`http://${baseIp}:${ollamaPort}/api/tags`);
              console.log(`Ollama response status: ${response.status}`);
              if (response.ok) {
                const data = await response.json();
                console.log("Ollama response data:", data);
                models = data.models.map((model) => model.name);
              } else {
                console.error(`Failed to fetch models from Ollama: ${response.status}`);
              }
            } else if (engine === "anthropic") {
              models = ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"];
            } else if (engine === "openai") {
              models = ["gpt-4-0125-preview", "gpt-4-1106-preview", "gpt-4-vision-preview", "gpt-4-1106-vision-preview", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106"];
            } 

            console.log("Fetched models:", models);
            modelWidget.options.values = models;
            console.log("Updated modelWidget.options.values:", modelWidget.options.values);
            
            // Update the selected_model value based on the available models
            if (models.includes(modelWidget.value)) {
              modelWidget.value = modelWidget.value;
            } else {
              modelWidget.value = models[0] || "";
            }

            this.triggerSlot(0);

          } catch (error) {
            console.error(`Error fetching models for engine ${engine}:`, error);
            modelWidget.options.values = [];
            modelWidget.value = "";
          }
        };
    
        engineWidget.callback = updateModels;
        baseIpWidget.callback = updateModels;
        ollamaPortWidget.callback = updateModels;

        // Initial update
        await updateModels();
      };
    }
  },
});
