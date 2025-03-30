import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import networkx as nx
import json
import os
import glob
from datetime import datetime
import numpy as np
import argparse


def load_network_data(log_file_path):
    """Load network data from a network.json file. Raises FileNotFoundError if not found."""
    if os.path.isdir(log_file_path):
        network_file = os.path.join(log_file_path, "network.json")
    else:
        network_file = log_file_path
        
    if not os.path.exists(network_file):
        # Raise error if file not found
        raise FileNotFoundError(f"Network log file not found: {network_file}")
        # return {"nodes": [], "edges": [], "sequence_log": [], "node_logs": {}, "edge_logs": {}}
    
    try:
        with open(network_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        # Still handle potential JSON errors
        print(f"Error decoding JSON from {network_file}: {e}")
        # Or re-raise a different error type if preferred
        raise ValueError(f"Error decoding JSON from {network_file}") from e 
        # return {"nodes": [], "edges": [], "sequence_log": [], "node_logs": {}, "edge_logs": {}}
    # except Exception as e:
    #     print(f"Error loading network data from {network_file}: {e}")
    #     return {"nodes": [], "edges": [], "sequence_log": [], "node_logs": {}, "edge_logs": {}}
    
    # Ensure essential keys exist, provide defaults if not (still useful)
    data.setdefault("nodes", [])
    data.setdefault("edges", [])
    data.setdefault("sequence_log", [])
    data.setdefault("node_logs", {})
    data.setdefault("edge_logs", {})
    
    return data


def load_spec_data(log_file_path):
    """Load spec data from a spec.json file if available"""
    if os.path.isdir(log_file_path):
        spec_file = os.path.join(log_file_path, "spec.json")
    else:
        spec_dir = os.path.dirname(log_file_path)
        spec_file = os.path.join(spec_dir, "spec.json")
        
    if not os.path.exists(spec_file):
        return None
    
    with open(spec_file, 'r') as f:
        data = json.load(f)
    
    return data


def create_network_graph(network_data, spec_data):
    """Create a NetworkX graph incorporating nodes and edges from spec and network data."""
    G = nx.DiGraph()
    node_types = {}

    # 1. Add nodes from spec_data if available
    if spec_data:
        # Add agents from spec
        if 'agents' in spec_data and isinstance(spec_data['agents'], list):
            for agent_spec in spec_data['agents']:
                if isinstance(agent_spec, dict) and 'name' in agent_spec:
                    node_id = agent_spec['name']
                    if node_id not in G:
                        G.add_node(node_id, type='agent')
                        node_types[node_id] = 'agent'

        # Add tools from spec
        if 'tools' in spec_data and isinstance(spec_data['tools'], list):
            for tool_spec in spec_data['tools']:
                 if isinstance(tool_spec, dict) and 'name' in tool_spec:
                    node_id = tool_spec['name']
                    if node_id not in G:
                        G.add_node(node_id, type='tool')
                        node_types[node_id] = 'tool'

        # Potentially add human node if defined in spec (adjust based on actual spec format)
        # Example: if spec_data.get('human_in_the_loop'):
        #     human_id = "User" # Or get ID from spec
        #     if human_id not in G:
        #         G.add_node(human_id, type='human')
        #         node_types[human_id] = 'human'

    # 2. Add/update nodes from network_data (network_data assumed to exist)
    for node in network_data['nodes']:
        node_id = node['id']
        node_type = node['type']
        if node_id not in G:
            G.add_node(node_id, type=node_type)
        # Update node type based on network data (it might be more specific or correct)
        G.nodes[node_id]['type'] = node_type
        node_types[node_id] = node_type

    # 3. Add potential edges from spec_data (agent -> tool connections)
    if spec_data and 'agents' in spec_data and isinstance(spec_data['agents'], list):
        for agent_spec in spec_data['agents']:
            if isinstance(agent_spec, dict) and 'name' in agent_spec and 'tools' in agent_spec:
                agent_id = agent_spec['name']
                if agent_id in G and isinstance(agent_spec['tools'], list):
                    for tool_name in agent_spec['tools']:
                        # Ensure the tool node also exists in the graph
                        if tool_name in G:
                            # Add edge with 'spec' source attribute
                            G.add_edge(agent_id, tool_name, edge_source='spec')
                        else:
                            print(f"Warning: Tool '{tool_name}' listed for agent '{agent_id}' not found as a node. Skipping spec edge.")
            

    # 4. Add actual communication edges from network_data
    for edge_data in network_data['edges']:
        from_node = edge_data['from']
        to_node = edge_data['to']
        # Ensure nodes exist before adding edge
        if from_node in G and to_node in G:
            # Add edge, potentially overwriting/updating 'spec' edge
            # Mark source as 'network' - this takes precedence if also in spec
            G.add_edge(from_node, to_node, edge_source='network') 
        else:
            print(f"Warning: Skipping edge from {from_node} to {to_node} because one or both nodes not found in graph.")

    return G, node_types


def get_node_positions(G):
    """Get node positions using Fruchterman-Reingold layout"""
    return nx.spring_layout(G, seed=42)


def create_network_animation(G, node_types, sequence_log, pos=None):
    """Create a network animation from the sequence log"""
    if pos is None:
        pos = get_node_positions(G)

    # Determine the set of nodes active in the sequence log
    nodes_in_sequence = set()
    if sequence_log:
        for entry in sequence_log:
            if entry.get('from'):
                nodes_in_sequence.add(entry['from'])
            if entry.get('to'):
                nodes_in_sequence.add(entry['to'])
    
    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"Node: {node}<br>Type: {node_types.get(node, 'unknown')}")
        
        # Set node color and size based on type AND activity
        node_type = node_types.get(node)
        if node not in nodes_in_sequence:
             # Static color/size for inactive nodes
            node_color.append("gray")
            node_size.append(8)
        else:
            # Type-based color/size for active nodes
            if node_type == "human":
                node_color.append("blue")
                node_size.append(15)
            elif node_type == "agent":
                node_color.append("green")
                node_size.append(12)
            elif node_type == "tool":
                node_color.append("red")
                node_size.append(10)
            else:
                node_color.append("darkgray") # Slightly different gray for active-unknown
                node_size.append(8)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=1, color='#888')
        ),
        name='Nodes'
    )
    
    # Create edge traces
    edge_traces = []
    for edge in G.edges(data=True): # Get edge data attribute
        u, v, data = edge
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_source = data.get('edge_source', 'network') # Default to network if somehow missing
        
        # Style based on edge source (dash/width/opacity), color is uniform
        base_edge_color = '#888' # Uniform base color
        if edge_source == 'spec':
            line_style = dict(width=0.8, color=base_edge_color, dash='dot') # Thinner, dotted for spec-only
            edge_opacity = 0.4
        else: # 'network'
            line_style = dict(width=1, color=base_edge_color, dash='solid') # Solid for network
            edge_opacity = 0.5

        # Create arrow shape for directed edge
        edge_trace = go.Scatter(
            x=[x0, x1, None], 
            y=[y0, y1, None],
            mode='lines',
            line=line_style, # Use conditional style
            hoverinfo='text',
            text=f"{u} → {v}<br>(Source: {edge_source})", # Add source info to hover
            name=f"{u} → {v}",
            opacity=edge_opacity # Use conditional opacity
        )
        edge_traces.append(edge_trace)
    
    # Create the base figure (always contains nodes and static edges)
    fig = go.Figure(
        data=[node_trace] + edge_traces,
        layout=go.Layout(
            title=dict(
                text="Agent Network Communication",
                font=dict(size=16)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            # Sliders and updatemenus will be added below, remove initial empty lists
            # sliders=[], 
            # updatemenus=[] 
        )
    )
    
    # Create animation frames (assuming sequence_log exists)
    frames = []
    active_edges = set()
    # Keep track of nodes active *up to this frame* for correct base coloring on revert
    cumulative_active_nodes = set() 
    
    for i, entry in enumerate(sequence_log):
        from_node = entry.get('from')
        to_node = entry.get('to')
        current_edge_tuple = (from_node, to_node)
        msg_type = entry.get('type')
        
        # Update cumulative active nodes
        if from_node: cumulative_active_nodes.add(from_node)
        if to_node: cumulative_active_nodes.add(to_node)
        
        # Add edge to active set for this frame's highlighting
        if G.has_edge(from_node, to_node): 
            active_edges.add(current_edge_tuple)
        
        # Create updated node colors and sizes for the frame
        frame_node_colors = []
        frame_node_sizes = []
        frame_node_text = []  # Add step info to node text
        
        for node in G.nodes():
             # Determine base color/size based on whether node has EVER been active
            node_type = node_types.get(node)
            is_ever_active = node in cumulative_active_nodes
            
            if not is_ever_active:
                base_color = "gray"
                base_size = 8
            else:
                if node_type == "human":
                    base_color = "blue"
                    base_size = 15
                elif node_type == "agent":
                    base_color = "green"
                    base_size = 12
                elif node_type == "tool":
                    base_color = "red"
                    base_size = 10
                else:
                    base_color = "darkgray"
                    base_size = 8

            # Add step info to node text
            node_info = f"Node: {node}<br>Type: {node_types.get(node, 'unknown')}"
            if node == from_node or node == to_node: # Currently active in this step
                node_info += f"<br>Active in Step {i+1}"
            elif is_ever_active: # Previously active
                pass # No need to add extra text
            frame_node_text.append(node_info)
            
            # Apply highlighting for current step
            if node == from_node:
                frame_node_colors.append("orange")
                frame_node_sizes.append(base_size * 1.5)
            elif node == to_node:
                frame_node_colors.append("yellow")
                frame_node_sizes.append(base_size * 1.5)
            else:
                # Use the determined base color/size
                frame_node_colors.append(base_color)
                frame_node_sizes.append(base_size)
        
        # Create updated edge colors and widths for the frame
        frame_edge_traces = []
        for edge_tuple in G.edges(): # Iterate through graph edges (u, v)
            u, v = edge_tuple
            edge_data = G.get_edge_data(u, v)
            edge_source = edge_data.get('edge_source', 'network')
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            
            # Create edge text with step info
            edge_info = f"{u} → {v}<br>(Source: {edge_source})"
            
            # Define base style for this edge (used when inactive or previously active)
            base_edge_color = '#888' # Uniform base color
            if edge_source == 'spec':
                base_line_style = dict(width=0.8, color=base_edge_color, dash='dot')
                base_opacity = 0.4
            else: # network
                base_line_style = dict(width=1.0, color=base_edge_color, dash='solid')
                base_opacity = 0.8 # Make previously active network edges slightly more prominent than base

            # Apply highlighting/styling for the frame
            if edge_tuple == current_edge_tuple:
                # Currently active edge
                frame_line_style = dict(width=3, color='yellow', dash='solid')
                frame_opacity = 1.0
                edge_info += f"<br>Active in Step {i+1}"
            elif edge_tuple in active_edges: # Previously active (must be network if in active_edges)
                 frame_line_style = base_line_style # Use network base style
                 frame_opacity = base_opacity 
            else:
                 # Inactive edge (spec or network)
                 # Use base style but dim opacity slightly more
                 if edge_source == 'spec':
                     frame_line_style = dict(width=0.8, color=base_edge_color, dash='dot')
                     frame_opacity = 0.3 # Dimmer spec
                 else:
                     frame_line_style = dict(width=1.0, color=base_edge_color, dash='solid')
                     frame_opacity = 0.4 # Dimmer inactive network
               
            edge_trace = go.Scatter(
                x=[x0, x1, None], 
                y=[y0, y1, None],
                mode='lines',
                line=frame_line_style,
                opacity=frame_opacity,
                hoverinfo='text',
                text=edge_info,
                customdata=[f"step-{i+1}"] 
            )
            frame_edge_traces.append(edge_trace)
        
        # Update node trace for this frame
        frame_node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                color=frame_node_colors,
                size=frame_node_sizes,
                line=dict(width=1, color='#888')
            ),
            text=frame_node_text,
            hoverinfo='text',
            customdata=[f"step-{i+1}"]
        )
        
        # Create the frame
        frame = go.Frame(
            data=[frame_node_trace] + frame_edge_traces,
            name=f"step_{i}",
            traces=list(range(len(frame_edge_traces) + 1))
        )
        frames.append(frame)
    
    # Add frames to figure
    fig.frames = frames
    
    # Add slider (assuming sequence_log exists)
    sliders = [dict(
        active=0,
        steps=[dict(
            method="animate",
            args=[[f"step_{k}"], dict(
                frame=dict(duration=300, redraw=True),
                mode="immediate",
                transition=dict(duration=0)
            )],
            label=f"Step {k+1}",
            value=k+1  # Store step number for tracking
        ) for k in range(len(sequence_log))],
        transition=dict(duration=0),
        x=0,
        y=0,
        currentvalue=dict(
            font=dict(size=12),
            prefix="Step: ",
            visible=True,
            xanchor="center"
        ),
        len=1.0
    )]
    
    # Add play/pause buttons (assuming sequence_log exists)
    updatemenus = [dict(
        type="buttons",
        buttons=[
            dict(
                label="Play",
                method="animate",
                args=[None, dict(frame=dict(duration=1000, redraw=True), fromcurrent=True)]
            ),
            dict(
                label="Pause",
                method="animate",
                args=[[None], dict(frame=dict(duration=0, redraw=True), mode="immediate")]
            )
        ],
        direction="left",
        pad=dict(r=10, t=10),
        showactive=False,
        x=0.1,
        xanchor="right",
        y=0,
        yanchor="top"
    )]
    
    # Update layout with animation controls
    fig.update_layout(sliders=sliders, updatemenus=updatemenus)
    
    return fig


def create_sequence_log_table(sequence_log):
    """Create a table from the sequence log"""
    rows = []
    for i, entry in enumerate(sequence_log):
        timestamp = entry.get('timestamp', '').split('T')[1].split('.')[0] if 'timestamp' in entry else ''
        from_node = entry.get('from', '')
        to_node = entry.get('to', '')
        msg_type = entry.get('type', '')
        
        # Format content based on message type
        if msg_type == 'tool_call' and 'arguments' in entry:
            content = str(entry.get('arguments', {}))
            if len(content) > 50:
                content = content[:50] + "..."
        else:
            content = entry.get('content', '')
            if content and len(content) > 50:
                content = content[:50] + "..."
        
        rows.append(html.Tr([
            html.Td(i+1),
            html.Td(timestamp),
            html.Td(from_node),
            html.Td(to_node),
            html.Td(msg_type),
            html.Td(content, id=f"step-content-{i+1}")
        ], id=f"log-row-{i+1}", className="log-row"))
    
    return html.Table([
        html.Thead(
            html.Tr([
                html.Th("Step"),
                html.Th("Time"),
                html.Th("From"),
                html.Th("To"),
                html.Th("Type"),
                html.Th("Content")
            ])
        ),
        html.Tbody(rows, id="log-table-body")
    ], className="table table-striped table-hover")


def create_subpath_list(spec_data):
    """Create a list of subpaths from the spec data"""
    if not spec_data or 'verification' not in spec_data or 'subpaths' not in spec_data['verification']:
        return html.Div("No subpaths found in spec data")
    
    subpaths = spec_data['verification']['subpaths']
    items = []
    
    for i, subpath in enumerate(subpaths):
        items.append(html.Li([
            html.Span(f"Subpath {i+1}: "),
            html.Span(" → ".join(subpath))
        ], id=f"subpath-{i}", className="subpath-item"))
    
    return html.Ul(items, className="list-group")


def list_run_dirs(log_dir="logs"):
    """List all run directories in the logs directory"""
    if not os.path.exists(log_dir):
        return []
    
    run_dirs = []
    for d in sorted(glob.glob(os.path.join(log_dir, "*")), reverse=True):
        network_file = os.path.join(d, "network.json")
        spec_file = os.path.join(d, "spec.json")
        # Check for BOTH spec.json and network.json existence
        if os.path.isdir(d) and os.path.exists(spec_file) and os.path.exists(network_file):
            run_name = os.path.basename(d)
            # Try to parse timestamp and create a nice display name
            try:
                parts = run_name.split('_', 2)
                if len(parts) >= 3:
                    date_str = parts[0]
                    time_str = parts[1]
                    rest = parts[2]
                    timestamp = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                    display_name = f"{timestamp.strftime('%Y-%m-%d %H:%M')} - {rest}"
                else:
                    display_name = run_name
            except:
                display_name = run_name
            
            run_dirs.append({"path": d, "name": run_name, "display": display_name})
    
    return run_dirs


def create_app(log_dir="logs", initial_run=None):
    """Create the Dash application for network visualization"""
    # Load available runs
    runs = list_run_dirs(log_dir)
    
    if not runs:
        print(f"No run directories found in {log_dir}")
        return None
    
    # Use the first run if no initial run is specified
    if initial_run is None:
        initial_run = runs[0]["path"]
    elif not os.path.exists(initial_run):
        print(f"Specified run not found: {initial_run}")
        initial_run = runs[0]["path"]
    
    # Load initial data
    network_data = load_network_data(initial_run)
    spec_data = load_spec_data(initial_run)
    G, node_types = create_network_graph(network_data, spec_data)
    pos = get_node_positions(G)
    sequence_log = network_data.get('sequence_log', [])
    initial_figure = create_network_animation(G, node_types, sequence_log, pos)
    
    # Initialize app
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    # Create dropdown options
    dropdown_options = [
        {"label": run["display"], "value": run["path"]} 
        for run in runs
    ]
    
    
    # Create subpath list
    subpath_list = create_subpath_list(spec_data)
    
    # Hidden div to track current step
    current_step_div = html.Div(id="current-step", style={"display": "none"}, children="0")
    
    # Create app layout
    app.layout = html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Agent Network Visualization", className="mt-4 mb-4"),
                    dbc.Card([
                        dbc.CardHeader("Select Network Run"),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id="run-selector",
                                options=dropdown_options,
                                value=initial_run,
                                clearable=False,
                                className="mb-3",
                                searchable=True
                            )
                        ])
                    ], className="mb-4")
                ])
            ]),
            
            dbc.Row([
                # Left side: Network graph
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Network Communication Graph"),
                        dbc.CardBody([
                            dcc.Loading(
                                id="loading-graph",
                                type="circle",
                                children=[
                                    html.Div([
                                        dcc.Graph(
                                            id="network-graph",
                                            figure=initial_figure, # Use pre-created figure
                                            config={
                                                'displayModeBar': True,
                                                'scrollZoom': True,
                                                'displaylogo': False
                                            },
                                            style={'height': '600px'}
                                        ),
                                        # Controls no longer need conditional wrapper
                                        dbc.Checklist(
                                            options=[
                                                {"label": "Auto-update message details during animation", "value": 1}
                                            ],
                                            value=[1],
                                            id="auto-update-toggle",
                                            switch=True,
                                            className="mt-2"
                                        ),
                                        current_step_div
                                    ])
                                ]
                            )
                        ])
                    ])
                ], width=8),
                
                # Right side: Controls and information
                dbc.Col([
                    # Subpaths section
                    dbc.Card([
                        dbc.CardHeader("Verification Subpaths"),
                        dbc.CardBody([
                            html.Div(id="subpath-list", children=subpath_list)
                        ]),
                        dbc.CardFooter([
                            dbc.Button("Highlight Selected Subpath", id="highlight-subpath-btn", color="primary", className="mr-2"),
                            dbc.Button("Reset View", id="reset-view-btn", color="secondary")
                        ])
                    ], className="mb-4"),
                    
                    # Message details section
                    dbc.Card([
                        dbc.CardHeader([
                            "Current Message Details",
                            html.Span(id="message-details-step", className="float-right")
                        ]),
                        dbc.CardBody([
                            html.Div(id="message-details", className="message-container")
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Event Sequence Log"),
                        dbc.CardBody([
                            html.Div(
                                id="sequence-log-table",
                                className="sequence-log-container",
                                children=create_sequence_log_table(sequence_log)
                            )
                        ])
                    ])
                ])
            ])
        ], fluid=True),
        # Store for tracking animation state
        dcc.Store(id="animation-state", data={"current_step": 0, "playing": False}),
        # Interval for animation updates
        dcc.Interval(id="animation-interval", interval=500, disabled=True)
    ])
    
    # Define callbacks
    
    @app.callback(
        [Output("network-graph", "figure"),
         Output("subpath-list", "children"),
         Output("sequence-log-table", "children"),
         Output("animation-state", "data")],
        [Input("run-selector", "value")]
    )
    def update_run(selected_run):
        if not selected_run:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        # Load data (guaranteed to work if run is listed)
        network_data = load_network_data(selected_run)
        spec_data = load_spec_data(selected_run)
        G, node_types = create_network_graph(network_data, spec_data)
        pos = get_node_positions(G)
        sequence_log = network_data.get('sequence_log', [])
        
        # Create network figure
        fig = create_network_animation(G, node_types, sequence_log, pos)
        
        # Create subpath list
        subpath_list = create_subpath_list(spec_data)
        
        # Create sequence log table
        sequence_table = create_sequence_log_table(sequence_log)
        
        # Reset animation state
        animation_state = {"current_step": 0, "playing": False}
        
        return fig, subpath_list, sequence_table, animation_state
    
    @app.callback(
        Output("current-step", "children"),
        [Input("network-graph", "clickData"),
         Input("network-graph", "relayoutData"),
         Input("animation-interval", "n_intervals")],
        [State("animation-state", "data"),
         State("run-selector", "value"),
         State("current-step", "children")]
    )
    def update_current_step(clickData, relayoutData, n_intervals, animation_state, selected_run, current_step):
        ctx = dash.callback_context
        if not ctx.triggered or not selected_run:
            return dash.no_update
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        network_data = load_network_data(selected_run)
        max_steps = len(network_data.get('sequence_log', []))
        
        # Handle slider movement via relayout data (when user drags the slider)
        if trigger_id == "network-graph" and relayoutData:
            if 'slider.active' in relayoutData:
                step_index = relayoutData['slider.active']
                return str(step_index + 1)  # Add 1 to convert 0-based index to 1-based step
            
            # Check for play/pause buttons
            if 'updatemenus[0].active' in relayoutData:
                # Update the animation playing state
                animation_state['playing'] = (relayoutData['updatemenus[0].active'] == 0)
        
        # Handle animation interval ticks (for auto-stepping)
        elif trigger_id == "animation-interval" and animation_state.get('playing', False):
            try:
                current_step_num = int(current_step)
                next_step = current_step_num + 1
                if next_step <= max_steps:
                    return str(next_step)
                else:
                    return "1"  # Loop back to first step
            except ValueError:
                return "1"  # Default to first step if parsing fails
        
        # Handle direct node/edge clicks that include step info
        elif trigger_id == "network-graph" and clickData:
            try:
                for point in clickData['points']:
                    if 'customdata' in point and isinstance(point['customdata'], list) and len(point['customdata']) > 0:
                        custom_value = point['customdata'][0]  # Get the first element from the array
                        if isinstance(custom_value, str) and custom_value.startswith('step-'):
                            step_num = custom_value.split('-')[1]
                            return step_num
            except:
                pass
        
        return dash.no_update
    
    @app.callback(
        Output("animation-interval", "disabled"),
        [Input("network-graph", "relayoutData")],
        [State("animation-state", "data"), State("run-selector", "value")]
    )
    def toggle_animation(relayoutData, animation_state, selected_run):
        if not selected_run:
            return True
        
        if not relayoutData:
            return not animation_state.get('playing', False)
        
        # Check for play/pause buttons
        if 'updatemenus[0].active' in relayoutData:
            return relayoutData['updatemenus[0].active'] != 0  # Enabled when play button (index 0) is clicked
        
        return not animation_state.get('playing', False)
    
    @app.callback(
        [Output("message-details", "children"),
         Output("message-details-step", "children")],
        [Input("network-graph", "clickData"),
         Input("log-table-body", "n_clicks"),
         Input("current-step", "children")],
        [State("run-selector", "value"),
         State("auto-update-toggle", "value")]
    )
    def update_message_details(clickData, n_clicks, current_step, selected_run, auto_update):
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
        
        if not selected_run:
            return html.Div("Select a run to view details"), ""
        
        network_data = load_network_data(selected_run)
        sequence_log = network_data.get('sequence_log', [])
        
        # Update based on current step (from slider/animation) if auto-update is enabled AND sequence log exists
        if trigger_id == "current-step" and current_step and auto_update and 1 in auto_update:
            try:
                step_index = int(current_step) - 1  # Convert from 1-based step to 0-based index
                if 0 <= step_index < len(sequence_log):
                    log_entry = sequence_log[step_index]
                    from_node = log_entry.get('from', '')
                    to_node = log_entry.get('to', '')
                    msg_type = log_entry.get('type', '')
                    timestamp = log_entry.get('timestamp', '').split('T')[1].split('.')[0] if 'timestamp' in log_entry else ''
                    
                    content = [
                        html.H5(f"Message: {from_node} → {to_node}"),
                        html.H6(f"Type: {msg_type}"),
                        html.P(f"Time: {timestamp}"),
                        html.Hr(),
                    ]
                    
                    # Format content based on message type
                    if msg_type == 'tool_call' and 'arguments' in log_entry:
                        msg_content = json.dumps(log_entry.get('arguments', {}), indent=2)
                        content.append(html.H6("Arguments:"))
                        content.append(html.Pre(msg_content, style={'white-space': 'pre-wrap'}))
                    else:
                        msg_content = log_entry.get('content', '')
                        content.append(html.H6("Content:"))
                        content.append(html.Div(msg_content, style={'white-space': 'pre-wrap'}))
                    
                    return html.Div(content), f"Step {step_index + 1}"
            except Exception as e:
                print(f"Error updating message details: {e}")
                pass
        
        # Handle node/edge clicks
        if trigger_id == "network-graph" and clickData:
            point = clickData['points'][0]
            curve_number = point['curveNumber']
            
            if curve_number == 0:  # Node click
                node_text = point['text']
                node_id = node_text.split('<br>')[0].replace('Node: ', '')
                
                # Get node logs
                node_logs = network_data.get('node_logs', {}).get(node_id, [])
                
                content = [
                    html.H5(f"Node: {node_id}"),
                    html.Hr(),
                    html.H6(f"Activity ({len(node_logs)} events):"),
                    html.Ul([
                        html.Li(f"Step {log.get('step')}: {log.get('type')} - {log.get('from')} → {log.get('to')}")
                        for log in sorted(node_logs, key=lambda x: x.get('step', 0))[:10]
                    ])
                ]
                
                if len(node_logs) > 10:
                    content.append(html.P(f"... and {len(node_logs) - 10} more events"))
                
                return html.Div(content), f"Node Info"
                
            else:  # Edge click
                edge_text = point['text']
                if not edge_text or '→' not in edge_text:
                    return html.Div("Click on a node or edge for details"), ""
                    
                from_node, to_node = edge_text.split('→')[0].strip(), edge_text.split('→')[1].strip().split('<br>')[0].strip()
                
                # Get edge logs
                edge_key = f"{from_node}→{to_node}"
                edge_logs = network_data.get('edge_logs', {}).get(edge_key, [])
                
                if not edge_logs:
                    return html.Div(f"No messages found between {from_node} and {to_node}"), "Edge Info"
                
                content = [
                    html.H5(f"Edge: {from_node} → {to_node}"),
                    html.Hr(),
                    html.H6(f"Messages ({len(edge_logs)} total):"),
                ]
                
                for i, log in enumerate(sorted(edge_logs, key=lambda x: x.get('step', 0))[:5]):
                    msg_type = log.get('type', '')
                    step_num = log.get('step', '?')
                    
                    if msg_type == 'tool_call' and 'arguments' in log:
                        msg_content = str(log.get('arguments', {}))
                    else:
                        msg_content = log.get('content', '')
                    
                    if len(msg_content) > 200:
                        msg_content = msg_content[:200] + "..."
                    
                    content.append(html.Div([
                        html.Strong(f"Step {step_num}: {msg_type}"),
                        html.P(msg_content)
                    ], className="message-item", id=f"edge-message-{step_num}"))
                
                if len(edge_logs) > 5:
                    content.append(html.P(f"... and {len(edge_logs) - 5} more messages"))
                
                return html.Div(content), f"Edge Info"
        
        # Handle log table row clicks
        if trigger_id == "log-table-body" and n_clicks:
            # Note: This would require additional work to capture which row was clicked
            # For now, we'll focus on the graph clicks and auto-updates
            pass
        
        # Simplify default message logic
        if trigger_id != "network-graph" and trigger_id != "log-table-body" and trigger_id != "current-step":
            return html.Div("Select a run or interact with the graph."), ""
        elif trigger_id == "network-graph" or trigger_id == "log-table-body" or trigger_id == "current-step":
             # Let the specific handlers return content or fall through
             pass

        # Fallback default if node/edge/log click didn't return anything
        return html.Div("Select a run or interact with the graph."), ""
    
    @app.callback(
        Output("network-graph", "figure", allow_duplicate=True),
        [Input("highlight-subpath-btn", "n_clicks")],
        [State("run-selector", "value")],
        prevent_initial_call=True
    )
    def highlight_subpath(n_clicks, selected_run):
        if not n_clicks or not selected_run:
            return dash.no_update
        
        # This is a placeholder - you would need to get the selected subpath
        # and highlight it in the graph. For now, we'll just return the current graph.
        return dash.no_update
    
    @app.callback(
        Output("network-graph", "figure", allow_duplicate=True),
        [Input("reset-view-btn", "n_clicks")],
        [State("run-selector", "value")],
        prevent_initial_call=True
    )
    def reset_view(n_clicks, selected_run):
        if not n_clicks or not selected_run:
            return dash.no_update
        
        # Reload the network graph
        network_data = load_network_data(selected_run)
        spec_data = load_spec_data(selected_run)
        G, node_types = create_network_graph(network_data, spec_data)
        pos = get_node_positions(G)
        sequence_log = network_data.get('sequence_log', [])
        
        return create_network_animation(G, node_types, sequence_log, pos)
    
    # Add custom CSS
    app.index_string = """
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>Agent Network Visualization</title>
            {%favicon%}
            {%css%}
            <style>
                .message-container {
                    max-height: 400px;
                    overflow-y: auto;
                }
                .sequence-log-container {
                    max-height: 300px;
                    overflow-y: auto;
                }
                .message-item {
                    margin-bottom: 15px;
                    padding-bottom: 10px;
                    border-bottom: 1px solid #eee;
                }
                .log-row:hover {
                    background-color: #f5f5f5;
                    cursor: pointer;
                }
                .active-row {
                    background-color: #e6f7ff !important;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    """
    
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent Network Visualization")
    parser.add_argument("--log-dir", "-d", default="logs", help="Path to the logs directory")
    parser.add_argument("--run", "-r", help="Specific run to visualize")
    parser.add_argument("--port", "-p", type=int, default=8050, help="Port to run the server on")
    args = parser.parse_args()
    
    app = create_app(args.log_dir, args.run)
    
    if app:
        print(f"Starting server on port {args.port}...")
        print(f"Open your browser to http://localhost:{args.port}")
        app.run(debug=True, port=args.port)
    else:
        print(f"No run directories found in {args.log_dir}")