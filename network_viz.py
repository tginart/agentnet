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
    """Load network data from a network.json file"""
    if os.path.isdir(log_file_path):
        network_file = os.path.join(log_file_path, "network.json")
    else:
        network_file = log_file_path
        
    if not os.path.exists(network_file):
        raise FileNotFoundError(f"Network log file not found: {network_file}")
    
    with open(network_file, 'r') as f:
        data = json.load(f)
    
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


def create_network_graph(network_data):
    """Create a NetworkX graph from the network data"""
    G = nx.DiGraph()
    
    # Add nodes with their types
    node_types = {}
    for node in network_data['nodes']:
        node_id = node['id']
        node_type = node['type']
        G.add_node(node_id, type=node_type)
        node_types[node_id] = node_type
    
    # Add edges
    for edge in network_data['edges']:
        G.add_edge(edge['from'], edge['to'])
    
    return G, node_types


def get_node_positions(G):
    """Get node positions using Fruchterman-Reingold layout"""
    return nx.spring_layout(G, seed=42)


def create_network_animation(G, node_types, sequence_log, pos=None):
    """Create a network animation from the sequence log"""
    if pos is None:
        pos = get_node_positions(G)
    
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
        
        # Set node color based on type
        if node_types.get(node) == "human":
            node_color.append("blue")
            node_size.append(15)
        elif node_types.get(node) == "agent":
            node_color.append("green")
            node_size.append(12)
        elif node_types.get(node) == "tool":
            node_color.append("red")
            node_size.append(10)
        else:
            node_color.append("gray")
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
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        # Create arrow shape for directed edge
        edge_trace = go.Scatter(
            x=[x0, x1, None], 
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=1, color='#888', dash='solid'),
            hoverinfo='text',
            text=f"{edge[0]} → {edge[1]}",
            name=f"{edge[0]} → {edge[1]}",
            opacity=0.5
        )
        edge_traces.append(edge_trace)
    
    # Create the figure
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
            height=600
        )
    )
    
    # Create animation frames
    frames = []
    active_edges = set()
    active_nodes = set()
    
    for i, entry in enumerate(sequence_log):
        from_node = entry.get('from')
        to_node = entry.get('to')
        msg_type = entry.get('type')
        
        # Add nodes to active set
        active_nodes.add(from_node)
        active_nodes.add(to_node)
        
        # Add edge to active set
        active_edges.add((from_node, to_node))
        
        # Create updated node colors and sizes
        frame_node_colors = []
        frame_node_sizes = []
        frame_node_text = []  # Add step info to node text
        
        for node in G.nodes():
            # Base colors and sizes
            if node_types.get(node) == "human":
                base_color = "blue"
                base_size = 15
            elif node_types.get(node) == "agent":
                base_color = "green"
                base_size = 12
            elif node_types.get(node) == "tool":
                base_color = "red"
                base_size = 10
            else:
                base_color = "gray"
                base_size = 8
            
            # Add step info to node text
            node_info = f"Node: {node}<br>Type: {node_types.get(node, 'unknown')}"
            if node in active_nodes:
                if node == from_node or node == to_node:
                    # Add current step info
                    node_info += f"<br>Active in Step {i+1}"
            frame_node_text.append(node_info)
            
            # Highlight active nodes
            if node in active_nodes:
                if node == from_node:
                    # Sender node
                    frame_node_colors.append("orange")
                    frame_node_sizes.append(base_size * 1.5)
                elif node == to_node:
                    # Receiver node
                    frame_node_colors.append("yellow")
                    frame_node_sizes.append(base_size * 1.5)
                else:
                    # Previously active node
                    frame_node_colors.append(base_color)
                    frame_node_sizes.append(base_size)
            else:
                frame_node_colors.append(base_color)
                frame_node_sizes.append(base_size)
        
        # Create updated edge colors and widths
        frame_edge_traces = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            # Create edge text with step info
            edge_info = f"{edge[0]} → {edge[1]}"
            
            if edge == (from_node, to_node):
                # Currently active edge
                edge_color = "yellow"
                edge_width = 3
                edge_opacity = 1.0
                edge_info += f"<br>Active in Step {i+1}"
            elif edge in active_edges:
                # Previously active edge
                edge_color = "#888"
                edge_width = 1.5
                edge_opacity = 0.8
            else:
                # Inactive edge
                edge_color = "#888"
                edge_width = 1
                edge_opacity = 0.3
            
            edge_trace = go.Scatter(
                x=[x0, x1, None], 
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=edge_width, color=edge_color),
                opacity=edge_opacity,
                hoverinfo='text',
                text=edge_info,
                customdata=[f"step-{i+1}"]  # Store step number as custom data in array format
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
            customdata=[f"step-{i+1}"]  # Store step number as custom data in array format
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
    
    # Add slider
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
    
    fig.update_layout(sliders=sliders)
    
    # Add play/pause buttons
    fig.update_layout(
        updatemenus=[dict(
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
    )
    
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
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "network.json")):
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
    G, node_types = create_network_graph(network_data)
    pos = get_node_positions(G)
    sequence_log = network_data.get('sequence_log', [])
    
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
                                            figure=create_network_animation(G, node_types, sequence_log, pos),
                                            config={
                                                'displayModeBar': True,
                                                'scrollZoom': True,
                                                'displaylogo': False
                                            },
                                            style={'height': '600px'}
                                        ),
                                        # Add a button to toggle auto-update of message details
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
        
        # Load data for selected run
        network_data = load_network_data(selected_run)
        spec_data = load_spec_data(selected_run)
        G, node_types = create_network_graph(network_data)
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
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
        
        if not selected_run:
            return dash.no_update
        
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
        [State("animation-state", "data")]
    )
    def toggle_animation(relayoutData, animation_state):
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
        
        # Update based on current step (from slider/animation) if auto-update is enabled
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
        
        # Default message
        return html.Div("Click on a node or edge to see details"), ""
    
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
        G, node_types = create_network_graph(network_data)
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