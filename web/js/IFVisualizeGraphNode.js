import { app } from "/scripts/app.js";

app.registerExtension({
    name: "Comfy.IFVisualizeGraphNode",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "IF_VisualizeGraph") {
            console.log("Registering IF_VisualizeGraph node");

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                console.log("IF_VisualizeGraph node created");
                onNodeCreated?.apply(this, arguments);

                this.size = [400, 300]; // Set node size
                this.graphImage = null;
                this.errorMessage = null;
            };

            nodeType.prototype.onExecuted = function(message) {
                console.log("IF_VisualizeGraph node executed", JSON.stringify(message));

                if (message && message.ui) {
                    if (message.ui.graph) {
                        console.log("Graph data received:", typeof message.ui.graph, message.ui.graph.substring(0, 100) + "...");
                        this.renderGraph(message.ui.graph, message.ui.layout);
                    } else if (message.ui.error) {
                        console.error("Error received", message.ui.error);
                        this.errorMessage = message.ui.error;
                        this.graphImage = null;
                        this.setDirtyCanvas(true, true);
                    } else {
                        console.warn("No graph data or error received");
                        this.errorMessage = "No graph data received";
                        this.graphImage = null;
                        this.setDirtyCanvas(true, true);
                    }
                } else {
                    console.warn("No message.ui received");
                    this.errorMessage = "No ui data received";
                    this.graphImage = null;
                    this.setDirtyCanvas(true, true);
                }
            };

            nodeType.prototype.renderGraph = function(graphData, layout) {
                console.log("Rendering graph", typeof graphData, graphData.substring(0, 100) + "...", layout);
                const self = this;

                const renderGraph = () => {
                    try {
                        const graph = JSON.parse(graphData);
                        console.log("Parsed graph data:", graph);

                        if (!graph.nodes || !graph.links) {
                            throw new Error("Graph data is missing nodes or links.");
                        }

                        // Create canvas
                        const canvas = document.createElement('canvas');
                        const width = 400;
                        const height = 300;
                        canvas.width = width;
                        canvas.height = height;
                        const context = canvas.getContext('2d');

                        // Run the simulation
                        const simulation = d3.forceSimulation(graph.nodes)
                            .force("link", d3.forceLink(graph.links).id(d => d.id).distance(50))
                            .force("charge", d3.forceManyBody().strength(-100))
                            .stop();

                        simulation.tick(300);

                        // Scale positions to fit the canvas
                        const xExtent = d3.extent(graph.nodes, d => d.x);
                        const yExtent = d3.extent(graph.nodes, d => d.y);
                        const xScale = d3.scaleLinear().domain(xExtent).range([20, width - 20]);
                        const yScale = d3.scaleLinear().domain(yExtent).range([20, height - 20]);

                        // Clear the canvas
                        context.clearRect(0, 0, width, height);

                        // Draw links
                        context.strokeStyle = "#999";
                        context.lineWidth = 1;
                        context.beginPath();
                        graph.links.forEach(d => {
                            context.moveTo(xScale(d.source.x), yScale(d.source.y));
                            context.lineTo(xScale(d.target.x), yScale(d.target.y));
                        });
                        context.stroke();

                        // Draw nodes
                        context.fillStyle = "#69b3a2";
                        graph.nodes.forEach(d => {
                            context.beginPath();
                            context.arc(xScale(d.x), yScale(d.y), 5, 0, 2 * Math.PI);
                            context.fill();
                        });

                        // Convert canvas to image
                        const image = new Image();
                        image.src = canvas.toDataURL();
                        image.onload = function() {
                            console.log("Graph image loaded");
                            self.graphImage = image;
                            self.setDirtyCanvas(true, true);
                        };
                    } catch (error) {
                        console.error("Error rendering graph:", error);
                        self.errorMessage = `Error rendering graph: ${error.message}`;
                        self.graphImage = null;
                        self.setDirtyCanvas(true, true);
                    }
                };

                if (typeof d3 === 'undefined') {
                    console.log("D3.js is not loaded, loading it now.");
                    const script = document.createElement('script');
                    script.src = "https://d3js.org/d3.v7.min.js";
                    script.onload = renderGraph;
                    document.head.appendChild(script);
                } else {
                    renderGraph();
                }
            };

            nodeType.prototype.onDrawBackground = function(ctx) {
                console.log("onDrawBackground called");
                if (this.graphImage) {
                    ctx.drawImage(this.graphImage, 0, 0, this.size[0], this.size[1]);
                } else if (this.errorMessage) {
                    ctx.fillStyle = "#f00";
                    ctx.fillText(this.errorMessage, 10, 20);
                } else {
                    ctx.fillStyle = "#999";
                    ctx.fillText("Rendering graph...", 10, 20);
                }
            };
        }
    }
});