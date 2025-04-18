import dash
from dash import html, dcc, Input, Output, State, callback, ALL, MATCH
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


def create_network_animation(G, node_types, sequence_log, pos=None, highlight_subpath_nodes=None):
    """Create a network animation from the sequence log, optionally highlighting a subpath."""
    if pos is None:
        pos = get_node_positions(G)
        
    highlight_nodes = set(highlight_subpath_nodes) if highlight_subpath_nodes else set()
    # Determine highlight edges based on highlight_nodes
    highlight_edges = set()
    if highlight_nodes:
        for u, v in G.edges():
            if u in highlight_nodes and v in highlight_nodes:
                highlight_edges.add((u, v))

    # Determine the set of nodes active in the sequence log
    nodes_in_sequence = set()
    if sequence_log:
        for entry in sequence_log:
            if entry.get('from'):
                nodes_in_sequence.add(entry['from'])
            if entry.get('to'):
                nodes_in_sequence.add(entry['to'])
    
    # Create BASE node trace properties
    node_x = []
    node_y = []
    node_text = []
    base_node_color = []
    base_node_size = []
    base_node_border_color = []
    base_node_border_width = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"Node: {node}<br>Type: {node_types.get(node, 'unknown')}")
        
        # Determine BASE style properties
        node_type = node_types.get(node)
        is_active = node in nodes_in_sequence
        is_highlighted = node in highlight_nodes

        # Base Color (based on activity)
        if not is_active:
            color = "gray"
            size = 8
        else:
            # Type-based color/size for active nodes
            if node_type == "human":
                color = "blue"
                size = 15
            elif node_type == "agent":
                color = "green"
                size = 12
            elif node_type == "tool":
                color = "red"
                size = 10
            else:
                color = "darkgray"
                size = 8
        
        # Base Border (based on highlight)
        if is_highlighted:
            border_color = "purple"
            border_width = 2
            size += 2 # Slightly larger if highlighted
        else:
            border_color = "#888"
            border_width = 1
            
        base_node_color.append(color)
        base_node_size.append(size)
        base_node_border_color.append(border_color)
        base_node_border_width.append(border_width)
    
    # Create BASE node trace object
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            color=base_node_color,
            size=base_node_size,
            line=dict(width=base_node_border_width, color=base_node_border_color)
        ),
        name='Nodes'
    )
    
    # Create BASE edge traces
    edge_traces = []
    for edge in G.edges(data=True): 
        u, v, data = edge
        edge_tuple = (u, v)
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_source = data.get('edge_source', 'network')
        
        # Style based on edge source AND highlight
        base_edge_color = '#888'
        is_highlighted_edge = edge_tuple in highlight_edges

        if is_highlighted_edge:
            line_style = dict(width=1.5, color="purple", dash='solid') # Highlighted edges solid purple
            edge_opacity = 0.7
        elif edge_source == 'spec':
            line_style = dict(width=0.8, color=base_edge_color, dash='dot') 
            edge_opacity = 0.4
        else: # 'network' (not highlighted)
            line_style = dict(width=1, color=base_edge_color, dash='solid')
            edge_opacity = 0.5

        edge_trace = go.Scatter(
            x=[x0, x1, None], 
            y=[y0, y1, None],
            mode='lines',
            line=line_style,
            hoverinfo='text',
            text=f"{u} → {v}<br>(Source: {edge_source}){ '<br>HIGHLIGHTED' if is_highlighted_edge else ''}",
            name=f"{u} → {v}",
            opacity=edge_opacity
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
        
        # Create updated node properties for the frame
        frame_node_colors = []
        frame_node_sizes = []
        frame_node_border_colors = []
        frame_node_border_widths = []
        frame_node_text = []
        
        for node in G.nodes():
             # Determine BASE color/size/border for this frame (based on CUMULATIVE activity & highlight)
            node_type = node_types.get(node)
            is_ever_active = node in cumulative_active_nodes
            is_highlighted = node in highlight_nodes
            
            # Base Color for frame (based on cumulative activity)
            if not is_ever_active:
                current_base_color = "gray"
                current_base_size = 8
            else:
                if node_type == "human": current_base_color, current_base_size = "blue", 15
                elif node_type == "agent": current_base_color, current_base_size = "green", 12
                elif node_type == "tool": current_base_color, current_base_size = "red", 10
                else: current_base_color, current_base_size = "darkgray", 8
            
            # Base Border for frame (based on highlight)
            if is_highlighted:
                current_border_color = "purple"
                current_border_width = 2
                current_base_size += 2 # Slightly larger if highlighted
            else:
                current_border_color = "#888"
                current_border_width = 1

            # Apply ACTIVE step highlighting (overrides color, modifies size)
            frame_color = current_base_color
            frame_size = current_base_size
            if node == from_node:
                frame_color = "orange" 
                frame_size = current_base_size * 1.5
            elif node == to_node:
                frame_color = "yellow"
                frame_size = current_base_size * 1.5

            frame_node_colors.append(frame_color)
            frame_node_sizes.append(frame_size)
            frame_node_border_colors.append(current_border_color) # Border is independent of active step color
            frame_node_border_widths.append(current_border_width)
            
            # Add step info to node text
            node_info = f"Node: {node}<br>Type: {node_types.get(node, 'unknown')}"
            if node == from_node or node == to_node: # Currently active in this step
                node_info += f"<br>Active in Step {i+1}"
            elif is_ever_active: # Previously active
                pass # No need to add extra text
            frame_node_text.append(node_info)
        
        # Create updated edge colors and widths for the frame
        frame_edge_traces = []
        for edge_tuple in G.edges(): # Iterate through graph edges (u, v)
            u, v = edge_tuple
            edge_data = G.get_edge_data(u, v)
            edge_source = edge_data.get('edge_source', 'network')
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            
            # Create edge text with step info
            edge_info = f"{u} → {v}<br>(Source: {edge_source}){ '<br>HIGHLIGHTED' if edge_tuple in highlight_edges else ''}"
            
            # Define base style for this edge 
            base_edge_color = '#888' 
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
                 # Revert to base style (purple if highlighted, grey otherwise)
                 frame_line_style = base_line_style 
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
                line=dict(width=frame_node_border_widths, color=frame_node_border_colors) # Use frame border properties
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
                # Pass arguments including dynamic duration from the slider
                args=[None, {
                    "frame": {"duration": 1000, "redraw": True}, # Default duration here is less critical
                    "fromcurrent": True,
                    "mode": "immediate", # Ensure immediate start
                    # Reference the slider value using Plotly.js syntax if possible, 
                    # or handle via callback (more robust)
                    # Let's handle via callback for robustness
                }]
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


def create_sequence_log_table(sequence_log_with_indices):
    """Create a table from the sequence log (with original indices) with expandable rows."""
    table_rows = [] 
    
    # Truncation length
    TRUNCATE_LEN = 100

    # Handle both list of entries and list of (index, entry) tuples
    # Check the type of the first element to determine format
    if sequence_log_with_indices and isinstance(sequence_log_with_indices[0], tuple):
        log_iterable = sequence_log_with_indices
        using_indices = True
    else:
        # Wrap the original list with indices if not already provided
        log_iterable = enumerate(sequence_log_with_indices)
        using_indices = False

    for item in log_iterable:
        if using_indices:
            original_index, entry = item
        else:
            original_index, entry = item # item is (index, entry) from enumerate

        timestamp = entry.get('timestamp', '').split('T')[1].split('.')[0] if 'timestamp' in entry else ''
        from_node = entry.get('from', '')
        to_node = entry.get('to', '')
        msg_type = entry.get('type', '')
        
        # Get full content
        if msg_type == 'tool_call' and 'arguments' in entry:
            full_content = json.dumps(entry.get('arguments', {}), indent=2)
            content_type = "Arguments"
        else:
            full_content = str(entry.get('content', ''))
            content_type = "Content"
        
        # Create truncated content
        truncated_content = full_content
        is_truncated = False
        if len(truncated_content) > TRUNCATE_LEN:
            truncated_content = truncated_content[:TRUNCATE_LEN] + "..."
            is_truncated = True
            
        # Use original_index for IDs
        row_id = {'type': 'log-row', 'index': original_index}
        collapse_id = {'type': 'log-collapse', 'index': original_index}

        # Create the main row with truncated content
        # Use original_index + 1 for displayed step number
        main_row = html.Tr([
            html.Td(original_index + 1), 
            html.Td(timestamp),
            html.Td(from_node),
            html.Td(to_node),
            html.Td(msg_type),
            html.Td(truncated_content, className="log-content-truncated")
        ], id=row_id, className="log-row", n_clicks=0)
        
        # Create the collapsible row with full content (only if truncated)
        if is_truncated:
            collapsible_row = html.Tr([
                html.Td(colSpan=6, children=[
                    dbc.Collapse([
                        html.Div([
                            html.H6(f"Full {content_type}:", style={'margin-top': '10px'}),
                            html.Pre(full_content, className="log-content-full")
                        ], className="log-content-expanded-container")
                    ], id=collapse_id, is_open=False)
                ])
            ], className="log-row-expanded")
            table_rows.append(main_row)
            table_rows.append(collapsible_row)
        else:
             table_rows.append(main_row)

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
        html.Tbody(table_rows, id="log-table-body") 
    ], className="table table-striped table-hover")


def create_subpath_list(spec_data):
    """Create a list of subpaths from the spec data"""
    items = [] # Initialize items list

    # Add the "All subpaths" option first
    all_item_id = {'type': 'subpath-item', 'index': 'all'} # Special index for 'all'
    items.append(html.Li(
        "All subpaths", 
        id=all_item_id, 
        className="subpath-item", 
        n_clicks=0
    ))

    # Now add the individual subpaths
    if not spec_data or 'verification' not in spec_data or 'subpaths' not in spec_data['verification']:
        # Still return the 'All' item even if no specific paths exist
        if not items:
             return html.Div("No subpaths found in spec data")
        # else fall through to return the list containing just 'All'
        pass 
    else:
        subpaths = spec_data['verification']['subpaths']
        for i, subpath in enumerate(subpaths):
            # Use pattern-matching ID for clientside callback input
            item_id = {'type': 'subpath-item', 'index': i}
            items.append(html.Li([
                html.Span(f"Subpath {i+1}: "),
                html.Span(" → ".join(subpath))
            # Use the dictionary ID, add className for styling/selection
            ], id=item_id, className="subpath-item", n_clicks=0)) 
    
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
        # Store for selected subpath ID
        dcc.Store(id='selected-subpath-store', data=None),
        # Store for expanded log row index
        dcc.Store(id='expanded-log-row-store', data=None),
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
                                        # Remove the auto-update toggle as its callback is being removed
                                        # dbc.Checklist(
                                        #     options=[
                                        #         {"label": "Auto-update message details during animation", "value": 1}
                                        #     ],
                                        #     value=[1],
                                        #     id="auto-update-toggle",
                                        #     switch=True,
                                        #     className="mt-2"
                                        # ),
                                        current_step_div
                                    ])
                                ]
                            )
                        ])
                    ]),
                    # Add Speed Slider Row below the graph card
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                dbc.Label("Animation Speed (ms/step):", html_for="speed-slider", style={'margin-right': '10px'}),
                                html.Div(id="speed-slider-output", style={'display': 'inline-block', 'width': '50px', 'font-weight': 'bold'})
                            ], style={'display': 'inline-block'}),
                            dcc.Slider(
                                id='speed-slider',
                                min=100,      # 100ms (fast)
                                max=2000,     # 2000ms (2 seconds, slow)
                                step=100,     # Steps of 100ms
                                value=1000,   # Default to 1000ms (1 second)
                                marks={i: f'{i}ms' for i in range(100, 2001, 300)}, # Optional marks
                                tooltip={"placement": "bottom", "always_visible": False}
                            )
                        ], width=12)
                    ], className="mt-2 mb-4") # Add some margin
                ], width=8),
                
                # Right side: Controls and information
                dbc.Col([
                    # Subpaths section
                    dbc.Card([
                        dbc.CardHeader("Verification Subpaths"),
                        dbc.CardBody([
                            html.Div(id="subpath-list", children=subpath_list)
                        ], id="subpath-card-body"),
                        dbc.CardFooter([
                            dbc.Button("Highlight Selected Subpath", id="highlight-subpath-btn", color="primary", className="mr-2"),
                            dbc.Button("Reset View", id="reset-view-btn", color="secondary")
                        ])
                    ], className="mb-4"),
                    
                    # REMOVE Message details section
                    # dbc.Card([
                    #     dbc.CardHeader([
                    #         "Current Message Details",
                    #         html.Span(id="message-details-step", className="float-right")
                    #     ]),
                    #     dbc.CardBody([
                    #         html.Div(id="message-details", className="message-container")
                    #     ])
                    # ])
                ], width=4)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Event Sequence Log"),
                        dbc.CardBody([
                            # Add Filter Controls Row
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Step #:"),
                                    dbc.Input(id='filter-step-num', type='number', placeholder="Go to step...", min=1, step=1)
                                ], width=2), # Adjust width
                                dbc.Col([
                                    dbc.Label("From Node:"),
                                    dcc.Dropdown(id='filter-from-node', multi=True, placeholder="Filter...")
                                ], width=3), # Adjust width
                                dbc.Col([
                                    dbc.Label("To Node:"),
                                    dcc.Dropdown(id='filter-to-node', multi=True, placeholder="Filter...")
                                ], width=3), # Adjust width
                                dbc.Col([
                                    dbc.Label("Message Type:"),
                                    dcc.Dropdown(id='filter-msg-type', multi=True, placeholder="Filter...")
                                ], width=4) # Adjust width
                            ], className="mb-3"),
                            # Existing Table Div
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
         Output("animation-state", "data"),
         Output("filter-from-node", "options"),
         Output("filter-to-node", "options"),
         Output("filter-msg-type", "options")],
        [Input("run-selector", "value")]
    )
    def update_run(selected_run):
        if not selected_run:
            # Add no_updates for the new filter option outputs
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
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
        
        # Generate filter options from sequence log
        all_nodes = set()
        all_types = set()
        for entry in sequence_log:
            if entry.get('from'): all_nodes.add(entry['from'])
            if entry.get('to'): all_nodes.add(entry['to'])
            if entry.get('type'): all_types.add(entry['type'])
            
        node_options = sorted([{'label': n, 'value': n} for n in all_nodes], key=lambda x: x['label'])
        type_options = sorted([{'label': t, 'value': t} for t in all_types], key=lambda x: x['label'])
        
        # Adjust return values to include filter options
        return fig, subpath_list, sequence_table, animation_state, node_options, node_options, type_options
    
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
        Output("network-graph", "figure", allow_duplicate=True),
        [Input("highlight-subpath-btn", "n_clicks")],
        [State("run-selector", "value"),
         State('selected-subpath-store', 'data')],
        prevent_initial_call=True
    )
    def highlight_subpath(n_clicks, selected_run, selected_subpath_data):
        # selected_subpath_data is likely a JSON string like '{"type":"subpath-item","index":2}'
        if not n_clicks or not selected_run or not selected_subpath_data:
            return dash.no_update
        
        subpath_dict = None
        if isinstance(selected_subpath_data, str):
            try:
                # Parse the JSON string from the store
                subpath_dict = json.loads(selected_subpath_data)
            except json.JSONDecodeError:
                print(f"Error decoding subpath data from store: {selected_subpath_data}")
                return dash.no_update
        elif isinstance(selected_subpath_data, dict):
             # It might already be a dict on initial load or if client-side fails?
             subpath_dict = selected_subpath_data
        else:
            print(f"Unexpected subpath data type: {type(selected_subpath_data)}")
            return dash.no_update
            
        subpath_index = subpath_dict.get('index')
        if subpath_index is None:
            print(f"Could not find 'index' in subpath data: {subpath_dict}")
            return dash.no_update
            
        # Load data to get subpath definition and graph
        network_data = load_network_data(selected_run)
        spec_data = load_spec_data(selected_run)
        
        highlight_nodes_list = []
        try:
            if spec_data and 'verification' in spec_data and 'subpaths' in spec_data['verification']:
                subpaths = spec_data['verification']['subpaths']
                if subpath_index == 'all':
                    # Calculate the union of all nodes in all subpaths
                    all_subpath_nodes = set()
                    for path in subpaths:
                        all_subpath_nodes.update(path)
                    highlight_nodes_list = list(all_subpath_nodes)
                    if not highlight_nodes_list:
                        print("Warning: 'All subpaths' selected, but no nodes found in any subpath.")
                elif isinstance(subpath_index, int) and 0 <= subpath_index < len(subpaths):
                    # Get the specific subpath node list
                    highlight_nodes_list = subpaths[subpath_index]
                else:
                    print(f"Warning: Selected subpath index '{subpath_index}' is invalid.")
            else:
                print("Warning: Could not find subpaths in spec_data.")
        except Exception as e:
            print(f"Error retrieving subpath nodes: {e}")
            return dash.no_update

        if not highlight_nodes_list:
             return dash.no_update # No nodes found for path

        # Regenerate graph and figure with highlight info
        G, node_types = create_network_graph(network_data, spec_data)
        pos = get_node_positions(G)
        sequence_log = network_data.get('sequence_log', [])
        
        # Pass the list of nodes to highlight
        fig = create_network_animation(G, node_types, sequence_log, pos, highlight_subpath_nodes=highlight_nodes_list)
        
        return fig
    
    @app.callback(
        [Output("network-graph", "figure", allow_duplicate=True),
         Output('selected-subpath-store', 'data', allow_duplicate=True)],
        [Input("reset-view-btn", "n_clicks")],
        [State("run-selector", "value")],
        prevent_initial_call=True
    )
    def reset_view(n_clicks, selected_run):
        if not n_clicks or not selected_run:
            # Also return no_update for the store
            return dash.no_update, dash.no_update
        
        # Reload the network graph without highlight info
        network_data = load_network_data(selected_run)
        spec_data = load_spec_data(selected_run)
        G, node_types = create_network_graph(network_data, spec_data)
        pos = get_node_positions(G)
        sequence_log = network_data.get('sequence_log', [])
        
        fig = create_network_animation(G, node_types, sequence_log, pos) # No highlight args
        
        # Return the default figure and None to clear the selected subpath store
        return fig, None
    
    # Callback to filter sequence log table
    @app.callback(
        Output("sequence-log-table", "children", allow_duplicate=True),
        [Input("filter-from-node", "value"),
         Input("filter-to-node", "value"),
         Input("filter-msg-type", "value"),
         Input("filter-step-num", "value")],
        [State("run-selector", "value")],
        prevent_initial_call=True
    )
    def filter_log_table(selected_from, selected_to, selected_types, step_num, selected_run):
        if not selected_run:
            return dash.no_update
            
        network_data = load_network_data(selected_run)
        sequence_log = network_data.get('sequence_log', [])
        
        # Check if any filters are active
        filters_active = selected_from or selected_to or selected_types or step_num is not None
        
        if not filters_active:
            # No filters applied, return the original table
            return create_sequence_log_table(sequence_log)
        
        filtered_log = []
        for i, entry in enumerate(sequence_log):
            # Check step number first (most specific)
            original_step_num = i + 1 # Original 1-based step number
            if step_num is not None and original_step_num != step_num:
                continue # Skip if step number doesn't match
                
            # Check other filters if step number matched or wasn't specified
            match_from = not selected_from or entry.get('from') in selected_from
            match_to = not selected_to or entry.get('to') in selected_to
            match_type = not selected_types or entry.get('type') in selected_types
            
            if match_from and match_to and match_type:
                # Append tuple: (original_index, entry)
                filtered_log.append((i, entry))
                
        # If filtering by step number and nothing else, the result should be max 1 row
        # If other filters are active, result could be 0 or 1 row.
        
        return create_sequence_log_table(filtered_log)

    # Callback to handle log row expansion and toggle collapse
    @app.callback(
        [Output('expanded-log-row-store', 'data'),
         Output({'type': 'log-collapse', 'index': ALL}, 'is_open')], # Add output for collapse state
        Input({'type': 'log-row', 'index': ALL}, 'n_clicks'),
        [State('expanded-log-row-store', 'data'),
         State({'type': 'log-collapse', 'index': ALL}, 'id')], # Get IDs of ALL potential collapse components
        prevent_initial_call=True,
    )
    def update_expanded_log_row(n_clicks, current_expanded_index, collapse_ids):
        ctx = dash.callback_context
        triggered_id = ctx.triggered_id
        
        # Filter out None clicks (can happen on initial load or if n_clicks isn't set)
        valid_clicks = [click for click in n_clicks if click is not None]
        if not triggered_id or not valid_clicks:
            # No row clicked yet, ensure all are collapsed
            initial_states = [False] * len(collapse_ids)
            return current_expanded_index if current_expanded_index is not None else None, initial_states

        clicked_index = triggered_id['index']
        
        # Determine the new state for the store
        new_expanded_index = None
        if clicked_index != current_expanded_index:
            new_expanded_index = clicked_index # Expand the new one
        # If clicked == current, new_expanded_index remains None (collapse)
            
        # Determine the is_open state for ALL collapse components
        # Important: The order of collapse_ids corresponds to the order of Outputs for the ALL pattern
        is_open_states = [new_expanded_index == comp_id['index'] for comp_id in collapse_ids]
        
        return new_expanded_index, is_open_states
            
    # Callback to update animation speed display and potentially figure arguments
    @app.callback(
        [Output('speed-slider-output', 'children'),
         Output('network-graph', 'figure', allow_duplicate=True)],
        [Input('speed-slider', 'value'),
         Input('network-graph', 'figure')], # Trigger when figure updates (e.g., on run change)
        prevent_initial_call=True
    )
    def update_animation_speed(slider_value, current_figure):
        if not slider_value or not current_figure:
            return dash.no_update, dash.no_update

        # Update the speed display
        speed_text = f"{slider_value}"
        
        # Update the play button's arguments in the figure layout
        # Be careful modifying the figure dict directly
        new_figure = go.Figure(current_figure) # Create a copy to modify
        
        if new_figure.layout.updatemenus:
             # Find the play button (assuming it's the first button in the first menu)
             try:
                 play_button_args = new_figure.layout.updatemenus[0].buttons[0].args
                 if len(play_button_args) > 1 and isinstance(play_button_args[1], dict):
                     play_button_args[1]['frame']['duration'] = slider_value
                 else:
                     # Initialize args structure if needed (shouldn't be necessary based on previous edit)
                      play_button_args = [None, {"frame": {"duration": slider_value, "redraw": True}, "fromcurrent": True, "mode": "immediate"}]
                      new_figure.layout.updatemenus[0].buttons[0].args = play_button_args
             except (IndexError, KeyError, AttributeError) as e:
                  print(f"Error updating play button args: {e}. Skipping figure update.")
                  return speed_text, dash.no_update # Only update text if figure structure is unexpected

        return speed_text, new_figure

    # Add custom CSS including selection style AND subpath scrolling
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
                .subpath-item {
                    cursor: pointer;
                    padding: 5px;
                    border-radius: 3px;
                    margin-bottom: 2px; /* Add some space */
                }
                .subpath-item:hover {
                    background-color: #eee;
                }
                .selected-subpath {
                    font-weight: bold;
                    text-decoration: underline;
                    color: #0d6efd; /* Bootstrap primary blue */
                    background-color: #e7f1ff; /* Light blue background */
                }
                /* Style for the subpath card body */
                #subpath-card-body {
                    height: 550px; /* Adjust as needed to match graph height */
                    overflow-y: auto;
                }
                /* Style for expanded log row content */
                .log-content-expanded-container {
                    padding: 10px;
                    background-color: #f8f9fa; /* Light background */
                    border: 1px solid #dee2e6; /* Border */
                    border-radius: 4px;
                    margin-top: -1px; /* Overlap slightly with the row above */
                }
                .log-content-full {
                    max-height: 200px; /* Limit height and add scroll */
                    overflow-y: auto;
                    white-space: pre-wrap; /* Keep formatting */
                    word-break: break-all; /* Break long words */
                    background-color: white; /* White background for pre */
                    padding: 5px;
                    border: 1px solid #ced4da;
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
    
    # CLIENT-SIDE CALLBACK FOR SUBPATH SELECTION
    app.clientside_callback(
        """
        function(clicked_id, current_selection) {
            const triggered = dash_clientside.callback_context.triggered;
            // console.log("Clientside triggered:", triggered);
            // console.log("Clicked ID:", clicked_id, "Current Selection:", current_selection);
            
            if (!triggered || triggered.length === 0 || !triggered[0].prop_id.includes('.n_clicks')) {
                // Initial load or trigger wasn't a click on a subpath item
                // We still need to style based on the initial store value if any
                document.querySelectorAll('.subpath-item').forEach(item => {
                    if (item.id === current_selection) {
                        item.classList.add('selected-subpath');
                    } else {
                        item.classList.remove('selected-subpath');
                    }
                });
                return current_selection || dash_clientside.no_update; 
            }

            const new_selection_id = triggered[0].prop_id.split('.')[0];
            let new_store_value = new_selection_id;

            // Toggle selection: if clicking the already selected item, deselect it
            if (new_selection_id === current_selection) {
                new_store_value = null;
            }

            // Update class names
            document.querySelectorAll('.subpath-item').forEach(item => {
                if (item.id === new_store_value) {
                    item.classList.add('selected-subpath');
                } else {
                    item.classList.remove('selected-subpath');
                }
            });
            
            // console.log("New store value:", new_store_value);
            return new_store_value;
        }
        """,
        Output('selected-subpath-store', 'data'),
        [Input({'type': 'subpath-item', 'index': ALL}, 'n_clicks')], # Use pattern matching input
        [State('selected-subpath-store', 'data')]
    )

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