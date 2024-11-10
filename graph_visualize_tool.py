import networkx as nx
import json
import webbrowser
import os
import http.server
import socketserver
import threading
import subprocess
import sys
import platform
import shutil

def graphml_to_json(graphml_file):
    G = nx.read_graphml(graphml_file)
    data = nx.node_link_data(G)
    return json.dumps(data)

def create_html(json_data, html_path):
    json_data = json_data.replace('\\"', '')
    html_content = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Graph Visualization</title>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            body, html {
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
                overflow: hidden;
            }
            svg {
                width: 100%;
                height: 100%;
            }
            .links line {
                stroke: #999;
                stroke-opacity: 0.6;
            }
            .nodes circle {
                stroke: #fff;
                stroke-width: 1.5px;
            }
            .node-label {
                font-size: 12px;
                pointer-events: none;
            }
            .link-label {
                font-size: 10px;
                fill: #666;
                pointer-events: none;
                opacity: 0;
                transition: opacity 0.3s;
            }
            .link:hover .link-label {
                opacity: 1;
            }
            .tooltip {
                position: absolute;
                text-align: left;
                padding: 10px;
                font: 12px sans-serif;
                background: lightsteelblue;
                border: 0px;
                border-radius: 8px;
                pointer-events: none;
                opacity: 0;
                transition: opacity 0.3s;
                max-width: 300px;
            }
            .legend {
                position: absolute;
                top: 10px;
                right: 10px;
                background-color: rgba(255, 255, 255, 0.8);
                padding: 10px;
                border-radius: 5px;
            }
            .legend-item {
                margin: 5px 0;
            }
            .legend-color {
                display: inline-block;
                width: 20px;
                height: 20px;
                margin-right: 5px;
                vertical-align: middle;
            }
        </style>
    </head>
    <body>
        <svg></svg>
        <div class="tooltip"></div>
        <div class="legend"></div>
        <script>
            const graphData = JSON.parse('{json_data}');
            
            const svg = d3.select("svg"),
                width = window.innerWidth,
                height = window.innerHeight;

            svg.attr("viewBox", [0, 0, width, height]);

            const g = svg.append("g");

            const entityTypes = [...new Set(graphData.nodes.map(d => d.entity_type))];
            const color = d3.scaleOrdinal(d3.schemeCategory10).domain(entityTypes);

            const simulation = d3.forceSimulation(graphData.nodes)
                .force("link", d3.forceLink(graphData.links).id(d => d.id).distance(150))
                .force("charge", d3.forceManyBody().strength(-300))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("collide", d3.forceCollide().radius(30));

            const linkGroup = g.append("g")
                .attr("class", "links")
                .selectAll("g")
                .data(graphData.links)
                .enter().append("g")
                .attr("class", "link");

            const link = linkGroup.append("line")
                .attr("stroke-width", d => Math.sqrt(d.value));

            const linkLabel = linkGroup.append("text")
                .attr("class", "link-label")
                .text(d => d.description || "");

            const node = g.append("g")
                .attr("class", "nodes")
                .selectAll("circle")
                .data(graphData.nodes)
                .enter().append("circle")
                .attr("r", 5)
                .attr("fill", d => color(d.entity_type))
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended));

            const nodeLabel = g.append("g")
                .attr("class", "node-labels")
                .selectAll("text")
                .data(graphData.nodes)
                .enter().append("text")
                .attr("class", "node-label")
                .text(d => d.id);

            const tooltip = d3.select(".tooltip");

            node.on("mouseover", function(event, d) {
                tooltip.transition()
                    .duration(200)
                    .style("opacity", .9);
                tooltip.html(`<strong>${d.id}</strong><br>Entity Type: ${d.entity_type}<br>Description: ${d.description || "N/A"}`)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px");
            })
            .on("mouseout", function(d) {
                tooltip.transition()
                    .duration(500)
                    .style("opacity", 0);
            });

            const legend = d3.select(".legend");
            entityTypes.forEach(type => {
                legend.append("div")
                    .attr("class", "legend-item")
                    .html(`<span class="legend-color" style="background-color: ${color(type)}"></span>${type}`);
            });

            simulation
                .nodes(graphData.nodes)
                .on("tick", ticked);

            simulation.force("link")
                .links(graphData.links);

            function ticked() {
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);

                linkLabel
                    .attr("x", d => (d.source.x + d.target.x) / 2)
                    .attr("y", d => (d.source.y + d.target.y) / 2)
                    .attr("text-anchor", "middle")
                    .attr("dominant-baseline", "middle");

                node
                    .attr("cx", d => d.x)
                    .attr("cy", d => d.y);

                nodeLabel
                    .attr("x", d => d.x + 8)
                    .attr("y", d => d.y + 3);
            }

            function dragstarted(event) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                event.subject.fx = event.subject.x;
                event.subject.fy = event.subject.y;
            }

            function dragged(event) {
                event.subject.fx = event.x;
                event.subject.fy = event.y;
            }

            function dragended(event) {
                if (!event.active) simulation.alphaTarget(0);
                event.subject.fx = null;
                event.subject.fy = null;
            }

            const zoom = d3.zoom()
                .scaleExtent([0.1, 10])
                .on("zoom", zoomed);

            svg.call(zoom);

            function zoomed(event) {
                g.attr("transform", event.transform);
            }

        </script>
    </body>
    </html>
    '''.replace("{json_data}", json_data.replace("'", "\\'").replace("\n", ""))
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

def start_server(port=8189):
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Server started at http://localhost:{port}")
        httpd.serve_forever()

def visualize_graph(graphml_file):
    if not os.path.exists(graphml_file):
        print(f"GraphML file not found: {graphml_file}")
        return

    html_path = "graph_visualization.html"
    try:
        json_data = graphml_to_json(graphml_file)
        create_html(json_data, html_path)
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        return

    port = 8189
    try:
        server_thread = threading.Thread(target=start_server, args=(port,))
        server_thread.daemon = True
        server_thread.start()
    except OSError as err:
        if "Address already in use" in str(err):
            print(f"Port {port} is already in use. Trying a different port...")
            for new_port in range(port + 1, port + 10):  # Try the next few ports
                try:
                    server_thread = threading.Thread(target=start_server, args=(new_port,))
                    server_thread.daemon = True
                    server_thread.start()
                    port = new_port
                    print(f"Server started at http://localhost:{port}")
                    break  # Success, exit loop
                except OSError as e:
                    if "Address already in use" not in str(e):
                        raise  # Re-raise if it's not "Address in use" error.
            else:  # If no ports were available
                raise RuntimeError(f"Could not find an available port between {port+1} and {port+9}.") from err
        else:
            raise  # Re-raise the OSError if it's not about the address being in use

    # Open default browser
    try:
        webbrowser.open(f'http://localhost:{port}/{html_path}')
    except Exception as e:
        print(f"Error opening browser: {str(e)}")

    print("Graph visualization is ready. The browser should open automatically.")
    print(f"If the browser doesn't open, please visit http://localhost:{port}/{html_path}")
    print("The server will continue running in the background.")
    print("You can close the browser tab when you're done viewing the graph.")