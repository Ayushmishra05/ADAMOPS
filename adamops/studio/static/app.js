/**
 * AdamOps Studio — Canvas Application
 * 
 * Handles: node palette, drag-drop, canvas nodes, port connections,
 * properties panel, pipeline execution, and results display.
 */

// =============================================================================
// State
// =============================================================================

const state = {
    nodeTypes: [],              // Available node type definitions from server
    canvasNodes: {},            // Placed nodes: { id: { id, type, x, y, params, ... } }
    connections: [],            // [{ id, from_node, from_port, to_node, to_port }]
    selectedNode: null,         // Currently selected node id
    nextNodeId: 1,
    // Dragging state
    dragging: null,             // { nodeId, offsetX, offsetY }
    // Connection drawing state
    connecting: null,           // { fromNode, fromPort, startX, startY }
    // View transform
    transform: { x: 0, y: 0, scale: 1 },
    panning: false,
};

const CATEGORY_ICONS = {
    data: "📊",
    preprocessing: "🧹",
    feature_engineering: "⚙️",
    splitting: "✂️",
    models: "🤖",
    evaluation: "📈",
};

const CATEGORY_LABELS = {
    data: "Data",
    preprocessing: "Preprocessing",
    feature_engineering: "Feature Engineering",
    splitting: "Splitting",
    models: "Models",
    evaluation: "Evaluation",
};

// =============================================================================
// Initialization
// =============================================================================

document.addEventListener("DOMContentLoaded", () => {
    loadNodeTypes();
    setupEventListeners();
    setupCanvasView();
    setupKeyboardShortcuts();
    requestAnimationFrame(updateMinimap);
});

async function loadNodeTypes() {
    try {
        const resp = await fetch("/api/nodes");
        const data = await resp.json();
        state.nodeTypes = data.nodes;
        renderPalette();
    } catch (e) {
        console.error("Failed to load node types:", e);
    }
}

// =============================================================================
// Palette Rendering
// =============================================================================

function renderPalette() {
    const palette = document.getElementById("node-palette");
    palette.innerHTML = "";

    // Group by category
    const categories = {};
    state.nodeTypes.forEach(nt => {
        if (!categories[nt.category]) categories[nt.category] = [];
        categories[nt.category].push(nt);
    });

    const categoryOrder = ["data", "preprocessing", "feature_engineering", "splitting", "models", "evaluation"];

    categoryOrder.forEach(cat => {
        if (!categories[cat]) return;
        const section = document.createElement("div");
        section.className = "palette-category";
        section.dataset.category = cat;

        const title = document.createElement("div");
        title.className = "palette-category-title";
        title.innerHTML = `<span class="cat-dot cat-${cat}"></span>${CATEGORY_LABELS[cat]}`;
        section.appendChild(title);

        categories[cat].forEach(nt => {
            const el = document.createElement("div");
            el.className = "palette-node";
            el.dataset.nodeType = nt.id;
            el.innerHTML = `<span class="node-icon">${CATEGORY_ICONS[cat] || "📦"}</span>${nt.label}`;
            el.draggable = true;

            el.addEventListener("dragstart", (e) => {
                e.dataTransfer.setData("text/plain", nt.id);
                e.dataTransfer.effectAllowed = "copy";
            });

            section.appendChild(el);
        });

        palette.appendChild(section);
    });
}

// =============================================================================
// Search Filter
// =============================================================================

document.getElementById("node-search").addEventListener("input", (e) => {
    const q = e.target.value.toLowerCase();
    document.querySelectorAll(".palette-node").forEach(el => {
        const text = el.textContent.toLowerCase();
        el.style.display = text.includes(q) ? "" : "none";
    });
    document.querySelectorAll(".palette-category").forEach(cat => {
        const visible = cat.querySelectorAll('.palette-node[style=""], .palette-node:not([style])');
        cat.style.display = visible.length > 0 ? "" : "none";
    });
});

// =============================================================================
// Canvas Drop
// =============================================================================

function setupEventListeners() {
    const container = document.getElementById("canvas-container");

    container.addEventListener("dragover", (e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = "copy";
    });

    container.addEventListener("drop", (e) => {
        e.preventDefault();
        const typeId = e.dataTransfer.getData("text/plain");
        if (!typeId) return;

        const rect = container.getBoundingClientRect();
        // Adjust drop coordinates by the current pan/zoom transform
        const x = (e.clientX - rect.left - state.transform.x) / state.transform.scale - 90;
        const y = (e.clientY - rect.top - state.transform.y) / state.transform.scale - 20;

        addNodeToCanvas(typeId, x, y);
    });

    // Click on canvas background to deselect
    container.addEventListener("mousedown", (e) => {
        if (e.target === container || e.target.id === "canvas" || e.target.id === "canvas-hint") {
            selectNode(null);
        }
    });

    // Mouse move for dragging and connection drawing
    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseup", onMouseUp);

    // Buttons
    document.getElementById("btn-execute").addEventListener("click", executePipeline);
    document.getElementById("btn-clear").addEventListener("click", clearCanvas);
    document.getElementById("btn-save").addEventListener("click", savePipeline);
    document.getElementById("btn-load").addEventListener("click", () => document.getElementById("pipeline-upload").click());

    document.getElementById("pipeline-upload").addEventListener("change", loadPipeline);

    // Panel tabs
    document.querySelectorAll(".panel-tab").forEach(tab => {
        tab.addEventListener("click", () => {
            document.querySelectorAll(".panel-tab").forEach(t => t.classList.remove("active"));
            document.querySelectorAll(".tab-content").forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            document.getElementById("tab-" + tab.dataset.tab).classList.add("active");
        });
    });
}

// =============================================================================
// Add Node to Canvas
// =============================================================================

function addNodeToCanvas(typeId, x, y) {
    const nodeType = state.nodeTypes.find(nt => nt.id === typeId);
    if (!nodeType) return;

    const nodeId = "node_" + state.nextNodeId++;
    const nodeData = {
        id: nodeId,
        type: typeId,
        x: Math.max(0, x),
        y: Math.max(0, y),
        params: {},
    };

    // Set default params
    nodeType.params.forEach(p => {
        nodeData.params[p.name] = p.default !== null && p.default !== undefined ? String(p.default) : "";
    });

    state.canvasNodes[nodeId] = nodeData;
    renderCanvasNode(nodeId);
    selectNode(nodeId);

    // Hide hint
    const hint = document.getElementById("canvas-hint");
    if (hint) hint.style.display = "none";
}

// =============================================================================
// Render Canvas Node
// =============================================================================

function renderCanvasNode(nodeId) {
    const nd = state.canvasNodes[nodeId];
    const nt = state.nodeTypes.find(t => t.id === nd.type);
    if (!nt) return;

    // Remove existing element if re-rendering
    const existing = document.getElementById(nodeId);
    if (existing) existing.remove();

    const el = document.createElement("div");
    el.id = nodeId;
    el.className = "canvas-node";
    el.style.left = nd.x + "px";
    el.style.top = nd.y + "px";

    // Header
    const header = document.createElement("div");
    header.className = "node-header";
    header.innerHTML = `
        <div class="cat-indicator cat-${nt.category}"></div>
        <span class="node-title">${nt.label}</span>
        <span class="node-delete" data-node="${nodeId}">✕</span>
    `;
    el.appendChild(header);

    // Ports container
    const ports = document.createElement("div");
    ports.className = "node-ports";

    // Input ports
    nt.inputs.forEach(p => {
        const row = document.createElement("div");
        row.className = "node-port-row input-row";
        const dot = document.createElement("div");
        dot.className = "port-dot input-port";
        dot.dataset.node = nodeId;
        dot.dataset.port = p.name;
        dot.dataset.dtype = p.dtype;
        // Check if connected
        if (state.connections.some(c => c.to_node === nodeId && c.to_port === p.name)) {
            dot.classList.add("connected");
        }
        const label = document.createElement("span");
        label.className = "port-label";
        label.textContent = p.label;
        row.appendChild(dot);
        row.appendChild(label);
        ports.appendChild(row);
    });

    // Output ports
    nt.outputs.forEach(p => {
        const row = document.createElement("div");
        row.className = "node-port-row output-row";
        const label = document.createElement("span");
        label.className = "port-label";
        label.textContent = p.label;
        const dot = document.createElement("div");
        dot.className = "port-dot output-port";
        dot.dataset.node = nodeId;
        dot.dataset.port = p.name;
        dot.dataset.dtype = p.dtype;
        if (state.connections.some(c => c.from_node === nodeId && c.from_port === p.name)) {
            dot.classList.add("connected");
        }
        row.appendChild(label);
        row.appendChild(dot);
        ports.appendChild(row);
    });

    el.appendChild(ports);

    // Event listeners
    header.addEventListener("mousedown", (e) => {
        if (e.target.classList.contains("node-delete")) return;
        e.preventDefault();
        state.dragging = {
            nodeId,
            offsetX: (e.clientX - nd.x * state.transform.scale) - state.transform.x,
            offsetY: (e.clientY - nd.y * state.transform.scale) - state.transform.y,
        };
        selectNode(nodeId);
    });

    el.querySelector(".node-delete").addEventListener("click", () => deleteNode(nodeId));

    // Port click handlers
    el.querySelectorAll(".output-port").forEach(dot => {
        dot.addEventListener("mousedown", (e) => {
            e.stopPropagation();
            const rect = dot.getBoundingClientRect();
            const container = document.getElementById("canvas-container").getBoundingClientRect();
            // Account for scale/pan in line start pos
            state.connecting = {
                fromNode: dot.dataset.node,
                fromPort: dot.dataset.port,
                startX: (rect.left + rect.width / 2 - container.left - state.transform.x) / state.transform.scale,
                startY: (rect.top + rect.height / 2 - container.top - state.transform.y) / state.transform.scale,
            };
        });
    });

    el.querySelectorAll(".input-port").forEach(dot => {
        dot.addEventListener("mouseup", (e) => {
            e.stopPropagation();
            if (state.connecting) {
                createConnection(
                    state.connecting.fromNode, state.connecting.fromPort,
                    dot.dataset.node, dot.dataset.port
                );
                state.connecting = null;
                removePreviewLine();
            }
        });
    });

    el.addEventListener("mousedown", () => selectNode(nodeId));

    document.getElementById("canvas").appendChild(el);
}

// =============================================================================
// Node Drag
// =============================================================================

function onMouseMove(e) {
    if (state.panning) {
        state.transform.x += e.movementX;
        state.transform.y += e.movementY;
        applyTransform();
        updateMinimap();
        return;
    }

    if (state.dragging) {
        const nd = state.canvasNodes[state.dragging.nodeId];
        nd.x = (e.clientX - state.dragging.offsetX - state.transform.x) / state.transform.scale;
        nd.y = (e.clientY - state.dragging.offsetY - state.transform.y) / state.transform.scale;
        const el = document.getElementById(state.dragging.nodeId);
        if (el) {
            el.style.left = nd.x + "px";
            el.style.top = nd.y + "px";
        }
        renderConnections();
    }

    if (state.connecting) {
        const container = document.getElementById("canvas-container").getBoundingClientRect();
        const endX = (e.clientX - container.left - state.transform.x) / state.transform.scale;
        const endY = (e.clientY - container.top - state.transform.y) / state.transform.scale;
        drawPreviewLine(state.connecting.startX, state.connecting.startY, endX, endY);
    }
}

function onMouseUp(e) {
    if (state.panning) {
        state.panning = false;
        document.body.style.cursor = "default";
    }
    if (state.dragging) {
        state.dragging = null;
        updateMinimap();
    }
    if (state.connecting) {
        state.connecting = null;
        removePreviewLine();
    }
}

// =============================================================================
// Connections
// =============================================================================

function createConnection(fromNode, fromPort, toNode, toPort) {
    // Don't connect to self
    if (fromNode === toNode) return;

    // Remove existing connection to this input port
    state.connections = state.connections.filter(
        c => !(c.to_node === toNode && c.to_port === toPort)
    );

    const conn = {
        id: `conn_${fromNode}_${fromPort}_${toNode}_${toPort}`,
        from_node: fromNode,
        from_port: fromPort,
        to_node: toNode,
        to_port: toPort,
    };
    state.connections.push(conn);

    // Re-render affected nodes to update port dots
    renderCanvasNode(fromNode);
    renderCanvasNode(toNode);
    renderConnections();

    // Re-select if needed
    if (state.selectedNode) {
        document.getElementById(state.selectedNode)?.classList.add("selected");
    }
}

function renderConnections() {
    const svg = document.getElementById("connections-svg");
    // Remove old connection lines (keep preview)
    svg.querySelectorAll(".connection-line").forEach(el => el.remove());

    const containerRect = document.getElementById("canvas-container").getBoundingClientRect();

    state.connections.forEach(conn => {
        const fromEl = document.querySelector(`#${conn.from_node} .output-port[data-port="${conn.from_port}"]`);
        const toEl = document.querySelector(`#${conn.to_node} .input-port[data-port="${conn.to_port}"]`);
        if (!fromEl || !toEl) return;

        const fromRect = fromEl.getBoundingClientRect();
        const toRect = toEl.getBoundingClientRect();

        const x1 = fromRect.left + fromRect.width / 2 - containerRect.left;
        const y1 = fromRect.top + fromRect.height / 2 - containerRect.top;
        const x2 = toRect.left + toRect.width / 2 - containerRect.left;
        const y2 = toRect.top + toRect.height / 2 - containerRect.top;

        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("d", bezierPath(x1, y1, x2, y2));
        path.setAttribute("class", "connection-line");
        path.dataset.connId = conn.id;

        // Double-click to delete connection
        path.style.pointerEvents = "stroke";
        path.style.cursor = "pointer";
        path.addEventListener("dblclick", () => {
            state.connections = state.connections.filter(c => c.id !== conn.id);
            renderCanvasNode(conn.from_node);
            renderCanvasNode(conn.to_node);
            renderConnections();
            if (state.selectedNode) {
                document.getElementById(state.selectedNode)?.classList.add("selected");
            }
        });

        svg.appendChild(path);
    });
}

function bezierPath(x1, y1, x2, y2) {
    const dx = Math.abs(x2 - x1) * 0.5;
    return `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
}

function drawPreviewLine(x1, y1, x2, y2) {
    const svg = document.getElementById("connections-svg");
    let preview = svg.querySelector(".connection-preview");
    if (!preview) {
        preview = document.createElementNS("http://www.w3.org/2000/svg", "path");
        preview.setAttribute("class", "connection-preview");
        svg.appendChild(preview);
    }
    preview.setAttribute("d", bezierPath(x1, y1, x2, y2));
}

function removePreviewLine() {
    const preview = document.querySelector(".connection-preview");
    if (preview) preview.remove();
}

// =============================================================================
// Node Selection & Properties
// =============================================================================

function selectNode(nodeId) {
    // Deselect previous
    document.querySelectorAll(".canvas-node.selected").forEach(el => el.classList.remove("selected"));
    state.selectedNode = nodeId;

    if (!nodeId) {
        document.getElementById("properties-empty").style.display = "";
        document.getElementById("properties-form").style.display = "none";
        return;
    }

    document.getElementById(nodeId)?.classList.add("selected");
    renderPropertiesPanel(nodeId);
}

function renderPropertiesPanel(nodeId) {
    const nd = state.canvasNodes[nodeId];
    const nt = state.nodeTypes.find(t => t.id === nd.type);
    if (!nt) return;

    document.getElementById("properties-empty").style.display = "none";
    document.getElementById("properties-form").style.display = "";
    document.getElementById("prop-node-title").textContent = nt.label;
    document.getElementById("prop-node-desc").textContent = nt.description;

    const fields = document.getElementById("prop-fields");
    fields.innerHTML = "";

    nt.params.forEach(p => {
        const field = document.createElement("div");
        field.className = "prop-field";

        const label = document.createElement("label");
        label.textContent = p.label + (p.required ? " *" : "");
        field.appendChild(label);

        const currentVal = nd.params[p.name] !== undefined ? nd.params[p.name] : (p.default || "");

        if (p.dtype === "select" && p.options) {
            const select = document.createElement("select");
            p.options.forEach(opt => {
                const option = document.createElement("option");
                option.value = opt;
                option.textContent = opt;
                option.selected = String(currentVal) === String(opt);
                select.appendChild(option);
            });
            select.addEventListener("change", () => {
                nd.params[p.name] = select.value;
            });
            field.appendChild(select);

        } else if (p.dtype === "file") {
            const wrapper = document.createElement("div");
            wrapper.className = "file-input-wrapper";
            const input = document.createElement("input");
            input.type = "text";
            input.value = currentVal;
            input.placeholder = "File path or upload...";
            input.addEventListener("input", () => {
                nd.params[p.name] = input.value;
            });
            const btn = document.createElement("button");
            btn.className = "file-btn";
            btn.textContent = "Upload";
            btn.addEventListener("click", () => {
                const fileInput = document.getElementById("file-upload");
                fileInput.onchange = async () => {
                    const file = fileInput.files[0];
                    if (!file) return;
                    const formData = new FormData();
                    formData.append("file", file);
                    try {
                        const resp = await fetch("/api/upload", { method: "POST", body: formData });
                        const data = await resp.json();
                        input.value = data.filepath;
                        nd.params[p.name] = data.filepath;
                    } catch (e) {
                        console.error("Upload failed:", e);
                    }
                };
                fileInput.click();
            });
            wrapper.appendChild(input);
            wrapper.appendChild(btn);
            field.appendChild(wrapper);

        } else if (p.dtype === "number") {
            const input = document.createElement("input");
            input.type = "number";
            input.step = "any";
            input.value = currentVal;
            input.addEventListener("input", () => {
                nd.params[p.name] = input.value;
            });
            field.appendChild(input);

        } else if (p.dtype === "boolean") {
            const input = document.createElement("input");
            input.type = "checkbox";
            input.checked = currentVal === "true" || currentVal === true;
            input.addEventListener("change", () => {
                nd.params[p.name] = String(input.checked);
            });
            field.appendChild(input);

        } else {
            const input = document.createElement("input");
            input.type = "text";
            input.value = currentVal;
            input.placeholder = p.label;
            input.addEventListener("input", () => {
                nd.params[p.name] = input.value;
            });
            field.appendChild(input);
        }

        fields.appendChild(field);
    });
}

// =============================================================================
// Delete Node
// =============================================================================

function deleteNode(nodeId) {
    // Remove connections
    state.connections = state.connections.filter(
        c => c.from_node !== nodeId && c.to_node !== nodeId
    );

    // Remove from state
    delete state.canvasNodes[nodeId];

    // Remove DOM element
    document.getElementById(nodeId)?.remove();

    // Deselect
    if (state.selectedNode === nodeId) selectNode(null);

    // Re-render connections
    renderConnections();

    // Show hint if empty
    if (Object.keys(state.canvasNodes).length === 0) {
        document.getElementById("canvas-hint").style.display = "";
    }
}

// =============================================================================
// Clear Canvas
// =============================================================================

function clearCanvas() {
    if (Object.keys(state.canvasNodes).length === 0) return;
    if (!confirm("Clear all nodes and connections?")) return;

    Object.keys(state.canvasNodes).forEach(id => {
        document.getElementById(id)?.remove();
    });
    state.canvasNodes = {};
    state.connections = [];
    state.selectedNode = null;

    renderConnections();
    selectNode(null);
    document.getElementById("canvas-hint").style.display = "";
}

// =============================================================================
// Execute Pipeline
// =============================================================================

async function executePipeline() {
    const nodes = Object.values(state.canvasNodes).map(nd => ({
        id: nd.id,
        type: nd.type,
        params: nd.params,
    }));

    if (nodes.length === 0) {
        alert("Add some nodes to the canvas first!");
        return;
    }

    const connections = state.connections.map(c => ({
        from_node: c.from_node,
        from_port: c.from_port,
        to_node: c.to_node,
        to_port: c.to_port,
    }));

    // Update UI
    const btn = document.getElementById("btn-execute");
    const statusEl = document.getElementById("status-indicator");
    btn.classList.add("executing");
    btn.disabled = true;
    btn.innerHTML = '<span>⏳</span> Running...';
    statusEl.className = "status-running";
    statusEl.textContent = "Executing...";

    // Reset node statuses
    document.querySelectorAll(".canvas-node").forEach(el => {
        el.classList.remove("success", "error", "executing");
    });

    // Switch to results tab
    document.querySelectorAll(".panel-tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(t => t.classList.remove("active"));
    document.querySelector('.panel-tab[data-tab="results"]').classList.add("active");
    document.getElementById("tab-results").classList.add("active");

    try {
        const resp = await fetch("/api/execute", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ nodes, connections }),
        });
        const result = await resp.json();
        displayResults(result);

        // Update node visual statuses
        for (const [nodeId, status] of Object.entries(result.node_status || {})) {
            const el = document.getElementById(nodeId);
            if (el) {
                el.classList.remove("executing");
                el.classList.add(status === "success" ? "success" : "error");
            }
        }

        if (result.success) {
            statusEl.className = "status-success";
            statusEl.textContent = `Done in ${result.total_time}s`;
        } else {
            statusEl.className = "status-error";
            statusEl.textContent = "Failed";
        }

    } catch (e) {
        console.error("Execution failed:", e);
        statusEl.className = "status-error";
        statusEl.textContent = "Error";
        displayResults({ success: false, logs: ["Error: " + e.message], node_status: {} });
    }

    btn.classList.remove("executing");
    btn.disabled = false;
    btn.innerHTML = '<span>▶</span> Execute';
}

// =============================================================================
// Display Results
// =============================================================================

function displayResults(result) {
    document.getElementById("results-empty").style.display = "none";
    document.getElementById("results-content").style.display = "";

    const metricsEl = document.getElementById("results-metrics");
    const logsEl = document.getElementById("results-logs");
    const detailsEl = document.getElementById("results-details");

    // Metrics
    metricsEl.innerHTML = "";
    if (result.final_metrics && Object.keys(result.final_metrics).length > 0) {
        const section = document.createElement("div");
        section.className = "results-section";
        section.innerHTML = `<h4>${result.success ? "✅" : "❌"} Metrics</h4>`;

        const grid = document.createElement("div");
        grid.className = "metrics-grid";

        for (const [name, value] of Object.entries(result.final_metrics)) {
            const card = document.createElement("div");
            card.className = "metric-card";
            const displayVal = typeof value === "number" ? value.toFixed(4) : String(value);
            card.innerHTML = `
                <div class="metric-name">${name}</div>
                <div class="metric-value">${displayVal}</div>
            `;
            grid.appendChild(card);
        }

        section.appendChild(grid);
        metricsEl.appendChild(section);
    }

    // Logs
    logsEl.innerHTML = "";
    if (result.logs && result.logs.length > 0) {
        const section = document.createElement("div");
        section.className = "results-section";
        section.innerHTML = '<h4>📋 Execution Log</h4>';

        const logList = document.createElement("div");
        logList.className = "log-list";

        result.logs.forEach(line => {
            const div = document.createElement("div");
            let cls = "log-line";
            if (line.startsWith("✓")) cls = "log-success";
            else if (line.startsWith("✗") || line.includes("Error") || line.includes("failed")) cls = "log-error";
            else if (line.startsWith("▶")) cls = "log-running";
            div.className = cls;
            div.textContent = line;
            logList.appendChild(div);
        });

        section.appendChild(logList);
        logsEl.appendChild(section);
    }

    // Node details
    detailsEl.innerHTML = "";
    if (result.node_results && Object.keys(result.node_results).length > 0) {
        const section = document.createElement("div");
        section.className = "results-section";
        section.innerHTML = '<h4>🔍 Node Details</h4>';

        for (const [nodeId, outputs] of Object.entries(result.node_results)) {
            const nd = state.canvasNodes[nodeId];
            const nt = nd ? state.nodeTypes.find(t => t.id === nd.type) : null;
            const label = nt ? nt.label : nodeId;
            const status = result.node_status[nodeId] || "pending";
            const time = result.node_times?.[nodeId];

            const card = document.createElement("div");
            card.className = "node-result-card";

            let badgeClass = "badge-" + status;
            let detailHtml = "";

            for (const [key, val] of Object.entries(outputs)) {
                if (val && val.shape) {
                    detailHtml += `<div>${key}: ${val.type} ${JSON.stringify(val.shape)}</div>`;
                    if (val.columns) {
                        detailHtml += `<div>Columns: ${val.columns.slice(0, 8).join(", ")}${val.columns.length > 8 ? "..." : ""}</div>`;
                    }
                } else if (typeof val === "object" && val !== null && !val.type) {
                    // Metrics dict
                    for (const [mk, mv] of Object.entries(val)) {
                        const displayMv = typeof mv === "number" ? mv.toFixed(4) : mv;
                        detailHtml += `<div>${mk}: ${displayMv}</div>`;
                    }
                }
            }

            card.innerHTML = `
                <div class="nrc-title">
                    ${label}
                    <span class="badge ${badgeClass}">${status}</span>
                    ${time ? `<span style="color:var(--text-muted);font-size:10px;margin-left:auto">${time.toFixed(2)}s</span>` : ""}
                </div>
                <div class="nrc-detail">${detailHtml}</div>
            `;

            section.appendChild(card);
        }

        detailsEl.appendChild(section);
    }
}

// =============================================================================
// Pan & Zoom
// =============================================================================

function setupCanvasView() {
    const container = document.getElementById("canvas-container");
    
    // Middle click pan
    container.addEventListener("mousedown", (e) => {
        if (e.button === 1 || (e.button === 0 && e.altKey)) {
            e.preventDefault();
            state.panning = true;
            document.body.style.cursor = "grabbing";
        }
    });

    // Mouse wheel zoom
    container.addEventListener("wheel", (e) => {
        e.preventDefault();
        
        const zoomSpeed = 0.001;
        const delta = -e.deltaY * zoomSpeed;
        const oldScale = state.transform.scale;
        let newScale = oldScale + delta;
        
        newScale = Math.max(0.1, Math.min(newScale, 2)); // Limits: 10% to 200%
        
        const rect = container.getBoundingClientRect();
        const cursorX = e.clientX - rect.left;
        const cursorY = e.clientY - rect.top;
        
        // Zoom relative to cursor
        state.transform.x = cursorX - (cursorX - state.transform.x) * (newScale / oldScale);
        state.transform.y = cursorY - (cursorY - state.transform.y) * (newScale / oldScale);
        state.transform.scale = newScale;
        
        applyTransform();
        updateMinimap();
    });

    // Zoom buttons
    document.getElementById("btn-zoom-in").addEventListener("click", () => setZoom(state.transform.scale + 0.1));
    document.getElementById("btn-zoom-out").addEventListener("click", () => setZoom(state.transform.scale - 0.1));
    document.getElementById("btn-zoom-fit").addEventListener("click", zoomToFit);
}

function setZoom(newScale) {
    newScale = Math.max(0.1, Math.min(newScale, 2));
    const container = document.getElementById("canvas-container");
    const rect = container.getBoundingClientRect();
    
    // Zoom toward center
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    
    const oldScale = state.transform.scale;
    state.transform.x = centerX - (centerX - state.transform.x) * (newScale / oldScale);
    state.transform.y = centerY - (centerY - state.transform.y) * (newScale / oldScale);
    state.transform.scale = newScale;
    
    applyTransform();
    updateMinimap();
}

function zoomToFit() {
    const nodes = Object.values(state.canvasNodes);
    if (nodes.length === 0) {
        state.transform = { x: 0, y: 0, scale: 1 };
        applyTransform();
        updateMinimap();
        return;
    }
    
    let minX = Infinity, minY = Infinity;
    let maxX = -Infinity, maxY = -Infinity;
    
    nodes.forEach(n => {
        minX = Math.min(minX, n.x);
        minY = Math.min(minY, n.y);
        maxX = Math.max(maxX, n.x + 200); // approx node width
        maxY = Math.max(maxY, n.y + 100); // approx node height
    });
    
    // Add padding
    minX -= 50; minY -= 50;
    maxX += 50; maxY += 50;
    
    const contentW = maxX - minX;
    const contentH = maxY - minY;
    
    const container = document.getElementById("canvas-container");
    const rect = container.getBoundingClientRect();
    
    const scaleX = rect.width / contentW;
    const scaleY = rect.height / contentH;
    
    state.transform.scale = Math.max(0.1, Math.min(1, Math.min(scaleX, scaleY) * 0.9));
    
    // Center it
    state.transform.x = (rect.width - contentW * state.transform.scale) / 2 - minX * state.transform.scale;
    state.transform.y = (rect.height - contentH * state.transform.scale) / 2 - minY * state.transform.scale;
    
    applyTransform();
    updateMinimap();
}

function applyTransform() {
    const t = state.transform;
    const transformStr = `translate(${t.x}px, ${t.y}px) scale(${t.scale})`;
    
    document.getElementById("canvas").style.transform = transformStr;
    document.getElementById("connections-svg").style.transform = transformStr;
    
    document.getElementById("zoom-level").textContent = Math.round(t.scale * 100) + "%";
}

// =============================================================================
// Minimap
// =============================================================================

function updateMinimap() {
    const container = document.getElementById("minimap");
    if (!container) return;
    
    const canvas = document.getElementById("minimap-canvas");
    const ctx = canvas.getContext("2d");
    
    // Match resolution to styling
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const nodes = Object.values(state.canvasNodes);
    if (nodes.length === 0) {
        document.getElementById("minimap-viewport").style.display = "none";
        return;
    }
    
    document.getElementById("minimap-viewport").style.display = "block";
    
    // Find canvas bounds
    let minX = Infinity, minY = Infinity;
    let maxX = -Infinity, maxY = -Infinity;
    
    nodes.forEach(n => {
        minX = Math.min(minX, n.x);
        minY = Math.min(minY, n.y);
        maxX = Math.max(maxX, n.x + 200);
        maxY = Math.max(maxY, n.y + 100);
    });
    
    // Add generous padding for the minimap view area
    minX = Math.min(minX - 500, state.transform.x < 0 ? -state.transform.x / state.transform.scale : minX);
    minY = Math.min(minY - 500, state.transform.y < 0 ? -state.transform.y / state.transform.scale : minY);
    maxX = Math.max(maxX + 500, maxX);
    maxY = Math.max(maxY + 500, maxY);
    
    const mw = canvas.width;
    const mh = canvas.height;
    
    const scaleX = mw / (maxX - minX);
    const scaleY = mh / (maxY - minY);
    const mScale = Math.min(scaleX, scaleY);
    
    const offX = (mw - (maxX - minX) * mScale) / 2 - minX * mScale;
    const offY = (mh - (maxY - minY) * mScale) / 2 - minY * mScale;
    
    // Draw connections
    ctx.strokeStyle = "rgba(99, 102, 241, 0.4)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    state.connections.forEach(conn => {
        const fromNd = state.canvasNodes[conn.from_node];
        const toNd = state.canvasNodes[conn.to_node];
        if (!fromNd || !toNd) return;
        
        const x1 = offX + (fromNd.x + 200) * mScale;
        const y1 = offY + (fromNd.y + 50) * mScale;
        const x2 = offX + toNd.x * mScale;
        const y2 = offY + (toNd.y + 50) * mScale;
        
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
    });
    ctx.stroke();
    
    // Draw nodes
    nodes.forEach(n => {
        const nx = offX + n.x * mScale;
        const ny = offY + n.y * mScale;
        const nw = 200 * mScale;
        const nh = 100 * mScale;
        
        ctx.fillStyle = n.id === state.selectedNode ? "#6366f1" : "#1e293b";
        ctx.beginPath();
        ctx.roundRect(nx, ny, nw, nh, 2);
        ctx.fill();
    });
    
    // Update viewport box
    const viewContainer = document.getElementById("canvas-container").getBoundingClientRect();
    
    const viewportBox = document.getElementById("minimap-viewport");
    
    const vx = offX + (-state.transform.x / state.transform.scale) * mScale;
    const vy = offY + (-state.transform.y / state.transform.scale) * mScale;
    const vw = (viewContainer.width / state.transform.scale) * mScale;
    const vh = (viewContainer.height / state.transform.scale) * mScale;
    
    viewportBox.style.left = vx + "px";
    viewportBox.style.top = vy + "px";
    viewportBox.style.width = vw + "px";
    viewportBox.style.height = vh + "px";
}

// =============================================================================
// Save / Load / Shortcuts
// =============================================================================

function setupKeyboardShortcuts() {
    document.addEventListener("keydown", (e) => {
        // Ignore if typing in an input
        if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
        
        // Delete selected node
        if (e.key === "Delete" || e.key === "Backspace") {
            if (state.selectedNode) {
                deleteNode(state.selectedNode);
            }
        }
        
        // Ctrl/Cmd + S to Save
        if ((e.ctrlKey || e.metaKey) && e.key === "s") {
            e.preventDefault();
            savePipeline();
        }
        
        // Ctrl/Cmd + Enter to Execute
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            e.preventDefault();
            executePipeline();
        }
        
        // Zoom shortcuts
        if ((e.ctrlKey || e.metaKey) && (e.key === "=" || e.key === "+")) {
            e.preventDefault();
            setZoom(state.transform.scale + 0.1);
        }
        if ((e.ctrlKey || e.metaKey) && e.key === "-") {
            e.preventDefault();
            setZoom(state.transform.scale - 0.1);
        }
        if ((e.ctrlKey || e.metaKey) && e.key === "0") {
            e.preventDefault();
            zoomToFit();
        }
    });
}

function savePipeline() {
    const pipeline = {
        version: "1.0",
        transform: state.transform,
        nodes: Object.values(state.canvasNodes),
        connections: state.connections
    };
    
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(pipeline, null, 2));
    const dlAnchorElem = document.createElement('a');
    dlAnchorElem.setAttribute("href", dataStr);
    dlAnchorElem.setAttribute("download", "pipeline.adamops");
    dlAnchorElem.click();
    dlAnchorElem.remove();
}

function loadPipeline(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = function(evt) {
        try {
            const data = JSON.parse(evt.target.result);
            
            // Clear current
            Object.keys(state.canvasNodes).forEach(id => {
                document.getElementById(id)?.remove();
            });
            
            // Set state
            state.canvasNodes = {};
            state.connections = data.connections || [];
            
            let maxId = 0;
            data.nodes.forEach(n => {
                state.canvasNodes[n.id] = n;
                const match = n.id.match(/\d+/);
                if (match && parseInt(match[0]) > maxId) {
                    maxId = parseInt(match[0]);
                }
            });
            state.nextNodeId = maxId + 1;
            
            if (data.transform) {
                state.transform = data.transform;
            } else {
                state.transform = { x: 0, y: 0, scale: 1 };
            }
            
            // Render
            Object.keys(state.canvasNodes).forEach(renderCanvasNode);
            renderConnections();
            applyTransform();
            updateMinimap();
            
            document.getElementById("canvas-hint").style.display = data.nodes?.length ? "none" : "";
            selectNode(null);
            
        } catch (err) {
            console.error(err);
            alert("Failed to load pipeline file. Invalid JSON.");
        }
        
        // Reset input
        e.target.value = "";
    };
    reader.readAsText(file);
}

// =============================================================================
// Pan & Zoom
// =============================================================================

function setupCanvasView() {
    const container = document.getElementById("canvas-container");
    
    // Middle click pan
    container.addEventListener("mousedown", (e) => {
        if (e.button === 1 || (e.button === 0 && e.altKey)) {
            e.preventDefault();
            state.panning = true;
            document.body.style.cursor = "grabbing";
        }
    });

    // Mouse wheel zoom
    container.addEventListener("wheel", (e) => {
        e.preventDefault();
        
        const zoomSpeed = 0.001;
        const delta = -e.deltaY * zoomSpeed;
        const oldScale = state.transform.scale;
        let newScale = oldScale + delta;
        
        newScale = Math.max(0.1, Math.min(newScale, 2)); // Limits: 10% to 200%
        
        const rect = container.getBoundingClientRect();
        const cursorX = e.clientX - rect.left;
        const cursorY = e.clientY - rect.top;
        
        // Zoom relative to cursor
        state.transform.x = cursorX - (cursorX - state.transform.x) * (newScale / oldScale);
        state.transform.y = cursorY - (cursorY - state.transform.y) * (newScale / oldScale);
        state.transform.scale = newScale;
        
        applyTransform();
        updateMinimap();
    });

    // Zoom buttons
    document.getElementById("btn-zoom-in").addEventListener("click", () => setZoom(state.transform.scale + 0.1));
    document.getElementById("btn-zoom-out").addEventListener("click", () => setZoom(state.transform.scale - 0.1));
    document.getElementById("btn-zoom-fit").addEventListener("click", zoomToFit);
}

function setZoom(newScale) {
    newScale = Math.max(0.1, Math.min(newScale, 2));
    const container = document.getElementById("canvas-container");
    const rect = container.getBoundingClientRect();
    
    // Zoom toward center
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    
    const oldScale = state.transform.scale;
    state.transform.x = centerX - (centerX - state.transform.x) * (newScale / oldScale);
    state.transform.y = centerY - (centerY - state.transform.y) * (newScale / oldScale);
    state.transform.scale = newScale;
    
    applyTransform();
    updateMinimap();
}

function zoomToFit() {
    const nodes = Object.values(state.canvasNodes);
    if (nodes.length === 0) {
        state.transform = { x: 0, y: 0, scale: 1 };
        applyTransform();
        updateMinimap();
        return;
    }
    
    let minX = Infinity, minY = Infinity;
    let maxX = -Infinity, maxY = -Infinity;
    
    nodes.forEach(n => {
        minX = Math.min(minX, n.x);
        minY = Math.min(minY, n.y);
        maxX = Math.max(maxX, n.x + 200); // approx node width
        maxY = Math.max(maxY, n.y + 100); // approx node height
    });
    
    // Add padding
    minX -= 50; minY -= 50;
    maxX += 50; maxY += 50;
    
    const contentW = maxX - minX;
    const contentH = maxY - minY;
    
    const container = document.getElementById("canvas-container");
    const rect = container.getBoundingClientRect();
    
    const scaleX = rect.width / contentW;
    const scaleY = rect.height / contentH;
    
    state.transform.scale = Math.max(0.1, Math.min(1, Math.min(scaleX, scaleY) * 0.9));
    
    // Center it
    state.transform.x = (rect.width - contentW * state.transform.scale) / 2 - minX * state.transform.scale;
    state.transform.y = (rect.height - contentH * state.transform.scale) / 2 - minY * state.transform.scale;
    
    applyTransform();
    updateMinimap();
}

function applyTransform() {
    const t = state.transform;
    const transformStr = `translate(${t.x}px, ${t.y}px) scale(${t.scale})`;
    
    document.getElementById("canvas").style.transform = transformStr;
    document.getElementById("connections-svg").style.transform = transformStr;
    
    document.getElementById("zoom-level").textContent = Math.round(t.scale * 100) + "%";
}

// =============================================================================
// Minimap
// =============================================================================

function updateMinimap() {
    const container = document.getElementById("minimap");
    if (!container) return;
    
    const canvas = document.getElementById("minimap-canvas");
    const ctx = canvas.getContext("2d");
    
    // Match resolution to styling
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const nodes = Object.values(state.canvasNodes);
    if (nodes.length === 0) {
        document.getElementById("minimap-viewport").style.display = "none";
        return;
    }
    
    document.getElementById("minimap-viewport").style.display = "block";
    
    // Find canvas bounds
    let minX = Infinity, minY = Infinity;
    let maxX = -Infinity, maxY = -Infinity;
    
    nodes.forEach(n => {
        minX = Math.min(minX, n.x);
        minY = Math.min(minY, n.y);
        maxX = Math.max(maxX, n.x + 200);
        maxY = Math.max(maxY, n.y + 100);
    });
    
    // Add generous padding for the minimap view area
    minX = Math.min(minX - 500, state.transform.x < 0 ? -state.transform.x / state.transform.scale : minX);
    minY = Math.min(minY - 500, state.transform.y < 0 ? -state.transform.y / state.transform.scale : minY);
    maxX = Math.max(maxX + 500, maxX);
    maxY = Math.max(maxY + 500, maxY);
    
    const mw = canvas.width;
    const mh = canvas.height;
    
    const scaleX = mw / (maxX - minX);
    const scaleY = mh / (maxY - minY);
    const mScale = Math.min(scaleX, scaleY);
    
    const offX = (mw - (maxX - minX) * mScale) / 2 - minX * mScale;
    const offY = (mh - (maxY - minY) * mScale) / 2 - minY * mScale;
    
    // Draw connections
    ctx.strokeStyle = "rgba(99, 102, 241, 0.4)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    state.connections.forEach(conn => {
        const fromNd = state.canvasNodes[conn.from_node];
        const toNd = state.canvasNodes[conn.to_node];
        if (!fromNd || !toNd) return;
        
        const x1 = offX + (fromNd.x + 200) * mScale;
        const y1 = offY + (fromNd.y + 50) * mScale;
        const x2 = offX + toNd.x * mScale;
        const y2 = offY + (toNd.y + 50) * mScale;
        
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
    });
    ctx.stroke();
    
    // Draw nodes
    nodes.forEach(n => {
        const nx = offX + n.x * mScale;
        const ny = offY + n.y * mScale;
        const nw = 200 * mScale;
        const nh = 100 * mScale;
        
        ctx.fillStyle = n.id === state.selectedNode ? "#6366f1" : "#1e293b";
        ctx.beginPath();
        ctx.roundRect(nx, ny, nw, nh, 2);
        ctx.fill();
    });
    
    // Update viewport box
    const viewContainer = document.getElementById("canvas-container").getBoundingClientRect();
    
    const viewportBox = document.getElementById("minimap-viewport");
    
    const vx = offX + (-state.transform.x / state.transform.scale) * mScale;
    const vy = offY + (-state.transform.y / state.transform.scale) * mScale;
    const vw = (viewContainer.width / state.transform.scale) * mScale;
    const vh = (viewContainer.height / state.transform.scale) * mScale;
    
    viewportBox.style.left = vx + "px";
    viewportBox.style.top = vy + "px";
    viewportBox.style.width = vw + "px";
    viewportBox.style.height = vh + "px";
}

// =============================================================================
// Save / Load / Shortcuts
// =============================================================================

function setupKeyboardShortcuts() {
    document.addEventListener("keydown", (e) => {
        // Ignore if typing in an input
        if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
        
        // Delete selected node
        if (e.key === "Delete" || e.key === "Backspace") {
            if (state.selectedNode) {
                deleteNode(state.selectedNode);
            }
        }
        
        // Ctrl/Cmd + S to Save
        if ((e.ctrlKey || e.metaKey) && e.key === "s") {
            e.preventDefault();
            savePipeline();
        }
        
        // Ctrl/Cmd + Enter to Execute
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            e.preventDefault();
            executePipeline();
        }
        
        // Zoom shortcuts
        if ((e.ctrlKey || e.metaKey) && (e.key === "=" || e.key === "+")) {
            e.preventDefault();
            setZoom(state.transform.scale + 0.1);
        }
        if ((e.ctrlKey || e.metaKey) && e.key === "-") {
            e.preventDefault();
            setZoom(state.transform.scale - 0.1);
        }
        if ((e.ctrlKey || e.metaKey) && e.key === "0") {
            e.preventDefault();
            zoomToFit();
        }
    });
}

function savePipeline() {
    const pipeline = {
        version: "1.0",
        transform: state.transform,
        nodes: Object.values(state.canvasNodes),
        connections: state.connections
    };
    
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(pipeline, null, 2));
    const dlAnchorElem = document.createElement('a');
    dlAnchorElem.setAttribute("href", dataStr);
    dlAnchorElem.setAttribute("download", "pipeline.adamops");
    dlAnchorElem.click();
    dlAnchorElem.remove();
}

function loadPipeline(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = function(evt) {
        try {
            const data = JSON.parse(evt.target.result);
            
            // Clear current
            Object.keys(state.canvasNodes).forEach(id => {
                document.getElementById(id)?.remove();
            });
            
            // Set state
            state.canvasNodes = {};
            state.connections = data.connections || [];
            
            let maxId = 0;
            data.nodes.forEach(n => {
                state.canvasNodes[n.id] = n;
                const match = n.id.match(/\d+/);
                if (match && parseInt(match[0]) > maxId) {
                    maxId = parseInt(match[0]);
                }
            });
            state.nextNodeId = maxId + 1;
            
            if (data.transform) {
                state.transform = data.transform;
            } else {
                state.transform = { x: 0, y: 0, scale: 1 };
            }
            
            // Render
            Object.keys(state.canvasNodes).forEach(renderCanvasNode);
            renderConnections();
            applyTransform();
            updateMinimap();
            
            document.getElementById("canvas-hint").style.display = data.nodes?.length ? "none" : "";
            selectNode(null);
            
        } catch (err) {
            console.error(err);
            alert("Failed to load pipeline file. Invalid JSON.");
        }
        
        // Reset input
        e.target.value = "";
    };
    reader.readAsText(file);
}
