import json
import logging
import os
import datetime
import networkx as nx
from collections import defaultdict
import argparse
import yaml
import shutil
import glob
from tabulate import tabulate
import re
from typing import Dict, List, Optional, Tuple, Any

class NetworkLogger:
    def __init__(self, log_dir="logs", run_config=None, spec_name=None):
        # Create a timestamp for the run
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a unique run directory using timestamp and spec name
        run_name = f"{self.timestamp}"
        if spec_name:
            # Extract basename without extension if spec_name is a path
            if '/' in spec_name:
                spec_name = os.path.basename(spec_name)
            spec_name = os.path.splitext(spec_name)[0]
            run_name += f"_{spec_name}"
        
        # Create full run directory path
        self.run_dir = os.path.join(log_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Set paths for log files
        self.log_file = os.path.join(self.run_dir, f"agent_network.log")
        self.network_file = os.path.join(self.run_dir, f"network.json")
        
        # Save run configuration if provided
        if run_config:
            self.save_config(run_config)
            
        # Copy the spec file if provided and exists
        if spec_name and os.path.exists(spec_name):
            spec_dest = os.path.join(self.run_dir, "spec.json")
            shutil.copy2(spec_name, spec_dest)
        
        # Set up logger
        self.logger = logging.getLogger("agent_network")
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers to avoid duplicate logging
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        
        # Log initialization info
        self.logger.info(f"Initialized logging to directory: {self.run_dir}")
        
        # Network structure
        self.graph = nx.DiGraph()
        self.node_logs = defaultdict(list)
        self.edge_logs = defaultdict(list)
        
        # Add user/human as a special root node
        self.graph.add_node("human", type="human")
        self.node_logs["human"] = []
        
        # Sequential log for overall execution
        self.sequence_log = []
        self.step_counter = 0
    
    def save_config(self, config):
        """Save the run configuration as YAML"""
        config_file = os.path.join(self.run_dir, "config.yaml")
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return config_file
    
    def save_task_and_result(self, task, result):
        """Save the task and result of the simulation"""
        task_result_file = os.path.join(self.run_dir, "task_result.yaml")
        data = {
            "task": task,
            "result": result
        }
        with open(task_result_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        self.logger.info(f"Task and result saved to {task_result_file}")
        return task_result_file
    
    def log_message(self, from_node, to_node, message, message_type="message"):
        """Log a message being sent from one node to another"""
        self.step_counter += 1
        timestamp = datetime.datetime.now().isoformat()
        
        # Make sure nodes exist
        if from_node not in self.graph:
            self.graph.add_node(from_node, type="agent" if from_node != "human" else "human")
        if to_node not in self.graph:
            self.graph.add_node(to_node, type="agent")
        
        # Add edge if it doesn't exist
        if not self.graph.has_edge(from_node, to_node):
            self.graph.add_edge(from_node, to_node, messages=[])
        
        # Prepare log entry
        log_entry = {
            "step": self.step_counter,
            "timestamp": timestamp,
            "from": from_node,
            "to": to_node,
            "type": message_type,
            "content": message
        }
        
        # Add to sequence log
        self.sequence_log.append(log_entry)
        
        # Add to node logs
        self.node_logs[from_node].append(log_entry)
        self.node_logs[to_node].append(log_entry)
        
        # Add to edge log
        self.edge_logs[(from_node, to_node)].append(log_entry)
        
        # Log to file
        self.logger.info(f"[{message_type.upper()}] From: {from_node} | To: {to_node} | Content: {message[:200]}...")
    
    def log_tool_call(self, agent, tool_name, arguments):
        """Log a tool call by an agent"""
        self.step_counter += 1
        timestamp = datetime.datetime.now().isoformat()
        
        # Make sure nodes exist
        if agent not in self.graph:
            self.graph.add_node(agent, type="agent")
        if tool_name not in self.graph:
            self.graph.add_node(tool_name, type="tool")
        
        # Add edge if it doesn't exist
        if not self.graph.has_edge(agent, tool_name):
            self.graph.add_edge(agent, tool_name, calls=[])
        
        # Prepare log entry
        log_entry = {
            "step": self.step_counter,
            "timestamp": timestamp,
            "from": agent,
            "to": tool_name,
            "type": "tool_call",
            "arguments": arguments
        }
        
        # Add to sequence log
        self.sequence_log.append(log_entry)
        
        # Add to node logs
        self.node_logs[agent].append(log_entry)
        self.node_logs[tool_name].append(log_entry)
        
        # Add to edge log
        self.edge_logs[(agent, tool_name)].append(log_entry)
        
        # Log to file
        self.logger.info(f"[TOOL_CALL] From: {agent} | To: {tool_name} | Args: {arguments}")
    
    def log_tool_result(self, tool_name, agent, result):
        """Log a tool result being returned to an agent"""
        self.step_counter += 1
        timestamp = datetime.datetime.now().isoformat()
        
        # Make sure nodes exist
        if agent not in self.graph:
            self.graph.add_node(agent, type="agent")
        if tool_name not in self.graph:
            self.graph.add_node(tool_name, type="tool")
        
        # Add edge if it doesn't exist
        if not self.graph.has_edge(tool_name, agent):
            self.graph.add_edge(tool_name, agent, results=[])
        
        # Prepare log entry
        log_entry = {
            "step": self.step_counter,
            "timestamp": timestamp,
            "from": tool_name,
            "to": agent,
            "type": "tool_result",
            "content": result
        }
        
        # Add to sequence log
        self.sequence_log.append(log_entry)
        
        # Add to node logs
        self.node_logs[tool_name].append(log_entry)
        self.node_logs[agent].append(log_entry)
        
        # Add to edge log
        self.edge_logs[(tool_name, agent)].append(log_entry)
        
        # Log to file
        self.logger.info(f"[TOOL_RESULT] From: {tool_name} | To: {agent} | Result: {result[:200]}...")
    
    def save_network_structure(self):
        """Save the final network structure and logs to a JSON file"""
        network_data = {
            "nodes": [{"id": node, "type": data.get("type", "unknown")} for node, data in self.graph.nodes(data=True)],
            "edges": [{"from": u, "to": v} for u, v in self.graph.edges()],
            "node_logs": dict(self.node_logs),
            "edge_logs": {f"{u}→{v}": logs for (u, v), logs in self.edge_logs.items()},
            "sequence_log": self.sequence_log
        }
        
        # Save to file using the path defined in __init__
        with open(self.network_file, 'w') as f:
            json.dump(network_data, f, indent=2)
        
        self.logger.info(f"Network structure saved to {self.network_file}")
        return self.network_file

def analyze_network_logs(log_file):
    """Analyze the network logs to extract insights"""
    # If log_file is a directory, look for network.json file inside it
    if os.path.isdir(log_file):
        network_file = os.path.join(log_file, "network.json")
    else:
        network_file = log_file
        
    if not os.path.exists(network_file):
        raise FileNotFoundError(f"Network log file not found: {network_file}")
    
    with open(network_file, 'r') as f:
        data = json.load(f)
    
    # Create a graph from the data
    G = nx.DiGraph()
    
    # Add nodes
    for node in data['nodes']:
        G.add_node(node['id'], type=node['type'])
    
    # Add edges
    for edge in data['edges']:
        G.add_edge(edge['from'], edge['to'])
    
    # Basic analysis
    analysis = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'node_degrees': {node: G.degree(node) for node in G.nodes()},
        'in_degrees': {node: G.in_degree(node) for node in G.nodes()},
        'out_degrees': {node: G.out_degree(node) for node in G.nodes()},
        'central_nodes': list(sorted(nx.degree_centrality(G).items(), key=lambda x: x[1], reverse=True))[:5]
    }
    
    return analysis

def verify_network_trajectory(log_file, spec_file):
    """
    Verify if the network execution matches specified trajectories and edge requirements.
    
    Args:
        log_file: Path to the network log JSON file or directory containing network.json
        spec_file: Path to the specification JSON file with verification criteria
        
    Returns:
        dict: Verification results with details on which checks passed or failed
    """
    # Handle log_file as directory
    if os.path.isdir(log_file):
        network_file = os.path.join(log_file, "network.json")
    else:
        network_file = log_file
        
    if not os.path.exists(network_file):
        raise FileNotFoundError(f"Network log file not found: {network_file}")
    
    # Load network log data
    with open(network_file, 'r') as f:
        log_data = json.load(f)
    
    # Handle spec_file if it's a directory
    if os.path.isdir(spec_file):
        spec_file = os.path.join(spec_file, "spec.json")
        
    if not os.path.exists(spec_file):
        raise FileNotFoundError(f"Specification file not found: {spec_file}")
    
    # Load specification with verification criteria
    with open(spec_file, 'r') as f:
        spec_data = json.load(f)
    
    # Get the verification criteria from spec
    verification = spec_data.get('verification', {})
    subpaths = verification.get('subpaths', [])
    edge_checks = verification.get('edge_checks', [])
    
    # Extract the actual trajectory from the log file
    sequence_log = log_data.get('sequence_log', [])
    
    # Build the actual path traversed as a sequence of nodes
    actual_path = []
    for entry in sequence_log:
        from_node = entry.get('from')
        to_node = entry.get('to')
        
        # Add nodes to the path if not already the last one
        if not actual_path or actual_path[-1] != from_node:
            actual_path.append(from_node)
        
        # Add destination node if not already the last one
        if not actual_path or actual_path[-1] != to_node:
            actual_path.append(to_node)
    
    # Initialize results
    results = {
        'subpath_checks': [],
        'edge_checks': [],
        'all_passed': True
    }
    
    # Verify subpaths (subsequence style)
    for i, subpath in enumerate(subpaths):
        result = check_subpath_exists(actual_path, subpath)
        results['subpath_checks'].append({
            'subpath': subpath,
            'passed': result,
            'details': f"Subpath {i+1}: {'Found' if result else 'Not found'} in trajectory"
        })
        
        if not result:
            results['all_passed'] = False
    
    # Verify edge checks
    for i, check in enumerate(edge_checks):
        check_type = check.get('check_type', 'assertEqual')
        from_node = check.get('from')
        to_node = check.get('to')
        msg_type = check.get('type')
        expected_args = check.get('arguments', {})
        
        result, details = verify_edge_check(log_data, check)
        results['edge_checks'].append({
            'check': check,
            'passed': result,
            'details': details
        })
        
        if not result:
            results['all_passed'] = False
    
    return results

def check_subpath_exists(actual_path, subpath):
    """
    Check if a subpath exists within the actual path using subsequence matching.
    A subsequence allows for non-consecutive elements but maintains order.
    
    Args:
        actual_path: List of nodes in the order they were visited
        subpath: Specified subpath that should exist in the actual path
        
    Returns:
        bool: True if subpath exists as a subsequence, False otherwise
    """
    if not subpath:
        return True  # Empty subpath always exists
    
    subpath_idx = 0  # Current index in subpath
    
    for node in actual_path:
        if node == subpath[subpath_idx]:
            subpath_idx += 1
            if subpath_idx == len(subpath):
                return True  # Found the complete subpath
    
    return False  # Subpath not found

def verify_edge_check(log_data, check):
    """
    Verify a specific edge check against the log data.
    
    Args:
        log_data: The network log JSON data
        check: The specific edge check to verify
        
    Returns:
        tuple: (passed, details) - Whether the check passed and details
    """
    check_type = check.get('check_type', 'assertEqual')
    from_node = check.get('from')
    to_node = check.get('to')
    msg_type = check.get('type')
    expected_args = check.get('arguments', {})
    
    # Get the edge key format
    edge_key = f"{from_node}→{to_node}"
    
    # Find logs for this edge
    edge_logs = log_data.get('edge_logs', {}).get(edge_key, [])
    
    # Filter by message type if specified
    if msg_type:
        edge_logs = [log for log in edge_logs if log.get('type') == msg_type]
    
    if not edge_logs:
        return False, f"No matching logs found for edge {from_node} → {to_node} with type {msg_type}"
    
    # For assertEqual check (default)
    if check_type == 'assertEqual':
        for log in edge_logs:
            # Check arguments for tool calls
            if msg_type == 'tool_call' and 'arguments' in log:
                args_match = True
                for key, expected_value in expected_args.items():
                    if key not in log['arguments'] or log['arguments'][key] != expected_value:
                        args_match = False
                        break
                
                if args_match:
                    return True, f"Found matching tool call from {from_node} to {to_node} with expected arguments"
            
            # Check content for other message types
            elif 'content' in log and log['content'] == check.get('content', ''):
                return True, f"Found matching message from {from_node} to {to_node} with expected content"
    
    # For contains check
    elif check_type == 'contains':
        for log in edge_logs:
            if msg_type == 'tool_call' and 'arguments' in log:
                args_match = True
                for key, expected_value in expected_args.items():
                    if key not in log['arguments'] or expected_value not in str(log['arguments'][key]):
                        args_match = False
                        break
                
                if args_match:
                    return True, f"Found tool call containing expected arguments pattern"
            
            elif 'content' in log and check.get('content', '') in log['content']:
                return True, f"Found message containing expected content pattern"
    
    # For exists check (just checks if the edge exists)
    elif check_type == 'exists':
        return True, f"Edge {from_node} → {to_node} exists"
    
    return False, f"No matching logs found for the specified criteria"

def visualize_network_ascii(log_file):
    """
    Visualize the network in ASCII/terminal format and allow interactive walkthrough of paths.
    
    Args:
        log_file: Path to the network log JSON file or directory containing network.json
    """
    # If log_file is a directory, look for network.json file inside it
    if os.path.isdir(log_file):
        network_file = os.path.join(log_file, "network.json")
    else:
        network_file = log_file
        
    if not os.path.exists(network_file):
        raise FileNotFoundError(f"Network log file not found: {network_file}")
    
    with open(network_file, 'r') as f:
        data = json.load(f)
    
    # Create a graph from the data
    G = nx.DiGraph()
    
    # Add nodes with their types
    node_types = {}
    for node in data['nodes']:
        node_id = node['id']
        node_type = node['type']
        G.add_node(node_id)
        node_types[node_id] = node_type
    
    # Add edges
    for edge in data['edges']:
        G.add_edge(edge['from'], edge['to'])
    
    # Get sequence logs and edge logs for reference
    sequence_log = data['sequence_log']
    
    # Node symbols based on type
    type_symbols = {
        "human": "👤",
        "agent": "🤖",
        "tool": "🔧",
        "unknown": "❓"
    }
    
    # ASCII art for node connections
    ascii_art = {
        "horizontal": "───",
        "vertical": "│",
        "corner_top_right": "┐",
        "corner_top_left": "┌",
        "corner_bottom_right": "┘",
        "corner_bottom_left": "└",
        "t_right": "├",
        "t_left": "┤",
        "t_up": "┴",
        "t_down": "┬",
        "cross": "┼"
    }
    
    def print_graph_overview():
        """Print an overview of the graph structure"""
        print("\n=== Network Graph Overview ===")
        print(f"Total nodes: {G.number_of_nodes()}")
        print(f"Total edges: {G.number_of_edges()}")
        print("\nNodes:")
        for node, node_type in node_types.items():
            symbol = type_symbols.get(node_type, "❓")
            print(f"  {symbol} {node} ({node_type})")
        
        print("\nConnections:")
        for u, v in G.edges():
            u_symbol = type_symbols.get(node_types.get(u, "unknown"), "❓")
            v_symbol = type_symbols.get(node_types.get(v, "unknown"), "❓")
            print(f"  {u_symbol} {u} → {v_symbol} {v}")
    
    def draw_ascii_path(path):
        """Draw an ASCII representation of the path"""
        if not path or len(path) < 2:
            print("No valid path to display")
            return
        
        path_str = ""
        for i, node in enumerate(path):
            node_symbol = type_symbols.get(node_types.get(node, "unknown"), "❓")
            if i < len(path) - 1:
                path_str += f"{node_symbol} {node} {ascii_art['horizontal']}> "
            else:
                path_str += f"{node_symbol} {node}"
        
        print("\nPath:")
        print(path_str)
    
    def display_message_on_path(current_node, next_node):
        """Display messages exchanged between current_node and next_node"""
        # Get the edge key format
        edge_key = f"{current_node}→{next_node}"
        
        # Find messages on this edge
        edge_logs = data['edge_logs'].get(edge_key, [])
        
        if not edge_logs:
            print(f"\nNo messages found between {current_node} and {next_node}")
            return
        
        print(f"\n=== Messages from {current_node} to {next_node} ===")
        for log in edge_logs:
            msg_type = log.get('type', 'unknown')
            step = log.get('step', '?')
            timestamp = log.get('timestamp', '?')
            
            # Format content based on message type
            if msg_type == 'tool_call':
                content = json.dumps(log.get('arguments', {}), indent=2)
                print(f"\n[Step {step}] {msg_type.upper()} at {timestamp}")
                print(f"Arguments: {content}")
            elif msg_type == 'tool_result':
                content = log.get('content', '')
                print(f"\n[Step {step}] {msg_type.upper()} at {timestamp}")
                print(f"Result: {content[:200]}{'...' if len(content) > 200 else ''}")
            else:
                content = log.get('content', '')
                print(f"\n[Step {step}] {msg_type.upper()} at {timestamp}")
                print(f"Content: {content[:200]}{'...' if len(content) > 200 else ''}")
            
            # Option to view full content
            if len(str(content)) > 200:
                choice = input("\nView full content? (y/n): ")
                if choice.lower() == 'y':
                    if msg_type == 'tool_call':
                        print(json.dumps(log.get('arguments', {}), indent=2))
                    else:
                        print(log.get('content', ''))
    
    def interactive_path_walk():
        """Allow user to interactively walk through paths in the graph"""
        current_node = None
        visited_path = []
        
        # Start with human node if available, otherwise let user choose
        if "human" in G.nodes():
            current_node = "human"
            visited_path.append(current_node)
        else:
            nodes = list(G.nodes())
            print("\nChoose a starting node:")
            for i, node in enumerate(nodes):
                node_symbol = type_symbols.get(node_types.get(node, "unknown"), "❓")
                print(f"{i+1}. {node_symbol} {node}")
            
            while current_node is None:
                try:
                    choice = int(input("\nEnter node number: "))
                    if 1 <= choice <= len(nodes):
                        current_node = nodes[choice-1]
                        visited_path.append(current_node)
                    else:
                        print("Invalid choice. Try again.")
                except ValueError:
                    print("Please enter a number.")
        
        while True:
            print("\n" + "=" * 50)
            print(f"Current node: {type_symbols.get(node_types.get(current_node, 'unknown'), '❓')} {current_node}")
            
            # Draw the current path
            draw_ascii_path(visited_path)
            
            # Get neighbors
            successors = list(G.successors(current_node))
            predecessors = list(G.predecessors(current_node))
            
            # Show options
            print("\nOptions:")
            print("0. Exit walk")
            print("1. View node details")
            
            # Forward connections
            for i, succ in enumerate(successors):
                node_symbol = type_symbols.get(node_types.get(succ, "unknown"), "❓")
                print(f"{i+2}. Go to {node_symbol} {succ}")
            
            # Backward connections (if not already included in options)
            back_start_idx = len(successors) + 2
            for i, pred in enumerate(predecessors):
                if pred not in successors:  # Avoid duplicates
                    node_symbol = type_symbols.get(node_types.get(pred, "unknown"), "❓")
                    print(f"{i+back_start_idx}. Go back to {node_symbol} {pred}")
            
            # Go back in visited path
            if len(visited_path) > 1:
                print(f"{len(successors) + len(predecessors) + 2}. Go back to previous node in path")
            
            # Get user choice
            try:
                choice = int(input("\nEnter your choice: "))
                
                if choice == 0:
                    break
                elif choice == 1:
                    # View node details
                    print(f"\n=== Node: {current_node} ===")
                    print(f"Type: {node_types.get(current_node, 'unknown')}")
                    print(f"In-degree: {G.in_degree(current_node)}")
                    print(f"Out-degree: {G.out_degree(current_node)}")
                    
                    # Show logs involving this node
                    node_logs = data.get('node_logs', {}).get(current_node, [])
                    print(f"\nActivity log ({len(node_logs)} entries):")
                    for i, log in enumerate(sorted(node_logs, key=lambda x: x.get('step', 0))):
                        step = log.get('step', '?')
                        log_type = log.get('type', 'unknown')
                        from_node = log.get('from', '?')
                        to_node = log.get('to', '?')
                        print(f"{i+1}. [Step {step}] {log_type} | {from_node} → {to_node}")
                    
                    # Option to view a specific log entry
                    if node_logs:
                        log_choice = input("\nView log entry (number) or press Enter to continue: ")
                        if log_choice.strip() and log_choice.isdigit():
                            log_idx = int(log_choice) - 1
                            if 0 <= log_idx < len(node_logs):
                                log = sorted(node_logs, key=lambda x: x.get('step', 0))[log_idx]
                                print("\n" + "=" * 40)
                                print(f"Step: {log.get('step')}")
                                print(f"Type: {log.get('type')}")
                                print(f"From: {log.get('from')} → To: {log.get('to')}")
                                if 'content' in log:
                                    print(f"Content: {log.get('content')}")
                                elif 'arguments' in log:
                                    print(f"Arguments: {json.dumps(log.get('arguments'), indent=2)}")
                                print("=" * 40)
                                input("Press Enter to continue...")
                
                elif 2 <= choice < back_start_idx:
                    # Go to successor
                    succ_idx = choice - 2
                    if 0 <= succ_idx < len(successors):
                        next_node = successors[succ_idx]
                        # Show messages on this path
                        display_message_on_path(current_node, next_node)
                        input("\nPress Enter to continue to next node...")
                        current_node = next_node
                        visited_path.append(current_node)
                
                elif back_start_idx <= choice < len(successors) + len(predecessors) + 2:
                    # Go to predecessor
                    pred_idx = choice - back_start_idx
                    if 0 <= pred_idx < len(predecessors):
                        if predecessors[pred_idx] not in successors:  # Skip if already in successors
                            next_node = predecessors[pred_idx]
                            # Show messages on this path
                            display_message_on_path(next_node, current_node)
                            input("\nPress Enter to continue to next node...")
                            current_node = next_node
                            visited_path.append(current_node)
                
                elif choice == len(successors) + len(predecessors) + 2 and len(visited_path) > 1:
                    # Go back in path
                    visited_path.pop()  # Remove current node
                    current_node = visited_path[-1]  # Go to previous node
                
                else:
                    print("Invalid choice. Try again.")
            
            except ValueError:
                print("Please enter a number.")
    
    # Main visualization function
    print("\n" + "=" * 60)
    print("ASCII Network Visualization and Interactive Path Walker")
    print("=" * 60)
    
    while True:
        print("\nMain Menu:")
        print("1. View network overview")
        print("2. Start interactive path walk")
        print("3. View sequence of events")
        print("4. Exit")
        
        choice = input("\nEnter your choice: ")
        
        if choice == '1':
            print_graph_overview()
        elif choice == '2':
            interactive_path_walk()
        elif choice == '3':
            # Display the sequence of events
            print("\n=== Sequence of Events ===")
            for i, log in enumerate(sequence_log):
                step = log.get('step', '?')
                log_type = log.get('type', 'unknown')
                from_node = log.get('from', '?')
                to_node = log.get('to', '?')
                from_symbol = type_symbols.get(node_types.get(from_node, "unknown"), "❓")
                to_symbol = type_symbols.get(node_types.get(to_node, "unknown"), "❓")
                
                print(f"{i+1}. [Step {step}] {log_type.upper()}")
                print(f"   {from_symbol} {from_node} → {to_symbol} {to_node}")
                
                # Show brief content summary based on type
                if log_type == 'tool_call':
                    args = log.get('arguments', {})
                    args_str = ", ".join([f"{k}: {str(v)[:30]}" for k, v in args.items()])
                    print(f"   Args: {args_str}")
                elif log_type in ['message', 'response', 'tool_result']:
                    content = log.get('content', '')
                    print(f"   Content: {content[:50]}{'...' if len(content) > 50 else ''}")
                
                # Add a separator between entries
                print()
            
            # Option to view full details of an entry
            view_choice = input("\nView full entry details (enter number) or press Enter to return: ")
            if view_choice.strip() and view_choice.isdigit():
                entry_idx = int(view_choice) - 1
                if 0 <= entry_idx < len(sequence_log):
                    log = sequence_log[entry_idx]
                    print("\n" + "=" * 60)
                    print(f"Step: {log.get('step')}")
                    print(f"Timestamp: {log.get('timestamp')}")
                    print(f"Type: {log.get('type')}")
                    print(f"From: {log.get('from')} → To: {log.get('to')}")
                    
                    if 'content' in log:
                        print("\nContent:")
                        print(log.get('content'))
                    elif 'arguments' in log:
                        print("\nArguments:")
                        print(json.dumps(log.get('arguments'), indent=2))
                    
                    print("=" * 60)
                    input("Press Enter to continue...")
        
        elif choice == '4':
            break
        else:
            print("Invalid choice. Try again.")

def get_run_info(run_dir: str) -> Dict[str, Any]:
    """
    Extract information about a run from its directory.
    
    Args:
        run_dir: Path to the run directory
        
    Returns:
        dict: Information about the run
    """
    info = {
        "path": run_dir,
        "name": os.path.basename(run_dir),
        "timestamp": None,
        "spec_name": None,
        "num_nodes": None,
        "num_edges": None,
        "task": None,
        "has_result": False,
        "duration": None,
        "model": None
    }
    
    # Extract timestamp and spec name from directory name using regex
    name = info["name"]
    
    # Try different patterns in order of specificity
    
    # Pattern 1: YYYYMMDD_HHMMSS_spec_name or YYYYMMDD_HHMMSS_numeric_spec_name
    pattern1 = re.compile(r'^(\d{8}_\d{6})_(?:\d+_)?(.+)$')
    match = pattern1.match(name)
    
    if match:
        # Extract timestamp and spec name
        timestamp_str = match.group(1)
        spec_name = match.group(2)
        
        try:
            info["timestamp"] = datetime.datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            info["spec_name"] = spec_name
        except ValueError:
            pass
    else:
        # Pattern 2: YYYYMMDD_spec_name (date only)
        pattern2 = re.compile(r'^(\d{8})_(?:\d+_)?(.+)$')
        match = pattern2.match(name)
        
        if match:
            date_str = match.group(1)
            spec_name = match.group(2)
            
            try:
                info["timestamp"] = datetime.datetime.strptime(date_str, "%Y%m%d")
                info["spec_name"] = spec_name
            except ValueError:
                pass
    
    # Check for network.json
    network_file = os.path.join(run_dir, "network.json")
    if os.path.exists(network_file):
        try:
            with open(network_file, 'r') as f:
                network_data = json.load(f)
                if 'nodes' in network_data:
                    info["num_nodes"] = len(network_data['nodes'])
                if 'edges' in network_data:
                    info["num_edges"] = len(network_data['edges'])
        except:
            pass
    
    # Check for config.yaml to get model information
    config_file = os.path.join(run_dir, "config.yaml")
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
                if 'model' in config_data:
                    info["model"] = config_data['model']
        except:
            pass
    
    # Check for task_result.yaml
    task_result_file = os.path.join(run_dir, "task_result.yaml")
    if os.path.exists(task_result_file):
        try:
            with open(task_result_file, 'r') as f:
                task_result_data = yaml.safe_load(f)
                if 'task' in task_result_data:
                    info["task"] = task_result_data['task']
                if 'result' in task_result_data:
                    info["has_result"] = True
        except:
            pass
    
    # Check log file for duration
    log_file = os.path.join(run_dir, "agent_network.log")
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    # Extract timestamps from first and last lines
                    first_ts = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', lines[0])
                    last_ts = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', lines[-1])
                    
                    if first_ts and last_ts:
                        start_time = datetime.datetime.strptime(first_ts.group(1), "%Y-%m-%d %H:%M:%S")
                        end_time = datetime.datetime.strptime(last_ts.group(1), "%Y-%m-%d %H:%M:%S")
                        info["duration"] = (end_time - start_time).total_seconds()
        except:
            pass
    
    return info

def list_runs(log_dir: str = "logs", only_complete: bool = True) -> List[Dict[str, Any]]:
    """
    List all run directories in the logs directory and extract information about each run.
    
    Args:
        log_dir: Path to the logs directory
        only_complete: If True, only include runs that have a network.json file
        
    Returns:
        list: List of run information dictionaries, sorted by timestamp (newest first)
    """
    # Ensure log directory exists
    if not os.path.exists(log_dir):
        return []
    
    # Get all directories in the logs directory
    run_dirs = [d for d in glob.glob(os.path.join(log_dir, "*")) if os.path.isdir(d)]
    
    # Filter for complete runs if requested
    if only_complete:
        complete_run_dirs = []
        for run_dir in run_dirs:
            network_file = os.path.join(run_dir, "network.json")
            if os.path.exists(network_file):
                complete_run_dirs.append(run_dir)
        run_dirs = complete_run_dirs
    
    # Extract information about each run
    runs = [get_run_info(run_dir) for run_dir in run_dirs]
    
    # Sort by timestamp (newest first)
    runs.sort(key=lambda x: x["timestamp"] if x["timestamp"] else datetime.datetime.min, reverse=True)
    
    return runs

def get_spec_info(spec_path: str) -> Dict[str, Any]:
    """
    Extract information about a network spec file.
    
    Args:
        spec_path: Path to the spec file
        
    Returns:
        dict: Information about the spec
    """
    info = {
        "path": spec_path,
        "name": os.path.basename(spec_path),
        "task": None,
        "num_agents": 0,
        "num_connections": 0
    }
    
    try:
        with open(spec_path, 'r') as f:
            spec_data = json.load(f)
            if 'task' in spec_data:
                info["task"] = spec_data['task']
            if 'agents' in spec_data:
                info["num_agents"] = len(spec_data['agents'])
            if 'connections' in spec_data:
                info["num_connections"] = len(spec_data['connections'])
    except:
        pass
    
    return info

def list_specs(specs_dir: str = None) -> List[Dict[str, Any]]:
    """
    List all spec files in the specs directory and extract information about each spec.
    
    Args:
        specs_dir: Path to the specs directory. If None, use the default path.
        
    Returns:
        list: List of spec information dictionaries
    """
    if specs_dir is None:
        # Try to find the specs directory relative to this file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        specs_dir = os.path.join(script_dir, "network_specs")
    
    # Ensure specs directory exists
    if not os.path.exists(specs_dir):
        return []
    
    # Get all JSON files in the specs directory
    spec_files = glob.glob(os.path.join(specs_dir, "*.json"))
    
    # Extract information about each spec
    specs = [get_spec_info(spec_file) for spec_file in spec_files]
    
    # Sort by name
    specs.sort(key=lambda x: x["name"])
    
    return specs

def find_matching_spec(spec_name: str, specs_dir: str = None) -> Optional[str]:
    """
    Find a spec file that matches the given name.
    
    Args:
        spec_name: Name or part of the name to match
        specs_dir: Path to the specs directory. If None, use the default path.
        
    Returns:
        str: Path to the matching spec file, or None if no match found
    """
    specs = list_specs(specs_dir)
    
    # First try exact match
    for spec in specs:
        if spec["name"] == spec_name or os.path.splitext(spec["name"])[0] == spec_name:
            return spec["path"]
    
    # Then try partial match
    for spec in specs:
        if spec_name.lower() in spec["name"].lower():
            return spec["path"]
    
    return None

def interactive_run_browser(log_dir: str = "logs", specs_dir: str = None, only_complete: bool = True):
    """
    Interactive terminal-based browser for past runs.
    
    Args:
        log_dir: Path to the logs directory
        specs_dir: Path to the specs directory (optional)
        only_complete: If True, only show complete runs initially
    """
    if specs_dir is None:
        # Try to find the specs directory relative to this file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        specs_dir = os.path.join(script_dir, "network_specs")
    
    # Get list of runs based on filter setting
    runs = list_runs(log_dir, only_complete=only_complete)
    
    if not runs:
        if only_complete:
            print(f"No complete runs found in {log_dir}. Checking for incomplete runs...")
            only_complete = False
            runs = list_runs(log_dir, only_complete=False)
            if runs:
                print(f"Found {len(runs)} incomplete runs.")
            else:
                print(f"No runs found in {log_dir}")
                sys.exit(1)
        else:
            print(f"No runs found in {log_dir}")
            sys.exit(1)
    
    selected_run = None
    page = 0
    page_size = 10
    
    def batch_operations():
        """Handle batch operations on filtered runs"""
        filtered_runs = runs  # Start with all runs
        current_filters = []  # Track active filters
        
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("\n" + "=" * 80)
            print(" " * 25 + "BATCH OPERATIONS")
            print("=" * 80)
            
            # Get unique values for filtering
            models = sorted(list(set(run["model"] for run in runs if run["model"])))
            specs = sorted(list(set(run["spec_name"] for run in runs if run["spec_name"])))
            
            # Show current filters if any
            if current_filters:
                print("\nActive Filters:")
                for f in current_filters:
                    print(f"  • {f}")
            else:
                print("\nNo active filters - showing all runs")
            print(f"Selected runs: {len(filtered_runs)}")
            
            # Show available filters
            print("\nAvailable Filters:")
            print("\nModels:")
            for i, model in enumerate(models):
                print(f"  {i+1}. {model}")
            
            print("\nSpecs:")
            for i, spec in enumerate(specs):
                print(f"  {i+1}. {spec}")
            
            print("\nFilter Options:")
            print("  1. Filter by model")
            print("  2. Filter by spec")
            print("  3. Clear filters")
            print("  4. Show current selection")
            print("  5. Run verification on selection")
            print("  6. Run analysis on selection")
            print("  7. Back to main menu")
            
            choice = input("\nEnter your choice: ").strip()
            
            if choice == '1':
                print("\nSelect model (enter number):")
                for i, model in enumerate(models):
                    print(f"  {i+1}. {model}")
                model_choice = input("\nEnter model number: ").strip()
                if model_choice.isdigit() and 0 < int(model_choice) <= len(models):
                    selected_model = models[int(model_choice)-1]
                    filtered_runs = [run for run in runs if run["model"] == selected_model]
                    current_filters = [f"Model: {selected_model}"]
                    print(f"\nSelected {len(filtered_runs)} runs with model {selected_model}")
                    input("\nPress Enter to continue...")
            
            elif choice == '2':
                print("\nSelect spec (enter number):")
                for i, spec in enumerate(specs):
                    print(f"  {i+1}. {spec}")
                spec_choice = input("\nEnter spec number: ").strip()
                if spec_choice.isdigit() and 0 < int(spec_choice) <= len(specs):
                    selected_spec = specs[int(spec_choice)-1]
                    filtered_runs = [run for run in runs if run["spec_name"] == selected_spec]
                    current_filters = [f"Spec: {selected_spec}"]
                    print(f"\nSelected {len(filtered_runs)} runs with spec {selected_spec}")
                    input("\nPress Enter to continue...")
            
            elif choice == '3':
                filtered_runs = runs
                current_filters = []
                print("\nFilters cleared.")
                input("Press Enter to continue...")
            
            elif choice == '4':
                print(f"\nCurrent selection: {len(filtered_runs)} runs")
                if current_filters:
                    print("Active filters:")
                    for f in current_filters:
                        print(f"  • {f}")
                else:
                    print("No active filters - showing all runs")
                
                # Create table data
                headers = ["#", "Date", "Spec", "Model", "Task", "Nodes", "Edges", "Duration", "Status"]
                table_data = []
                
                for i, run in enumerate(filtered_runs):
                    # Format timestamp
                    date_str = run["timestamp"].strftime("%Y-%m-%d %H:%M") if run["timestamp"] else "Unknown"
                    
                    # Format duration
                    if run["duration"] is not None:
                        if run["duration"] < 60:
                            duration_str = f"{run['duration']:.1f}s"
                        elif run["duration"] < 3600:
                            duration_str = f"{run['duration']/60:.1f}m"
                        else:
                            duration_str = f"{run['duration']/3600:.1f}h"
                    else:
                        duration_str = "N/A"
                    
                    # Format task (truncate if too long)
                    task = run["task"] if run["task"] else "N/A"
                    if task and len(task) > 30:
                        task = task[:27] + "..."
                    
                    # Format spec name (truncate if too long)
                    spec_name = run["spec_name"] or "Unknown"
                    if len(spec_name) > 15:
                        spec_name = spec_name[:12] + "..."
                    
                    # Format model (truncate if too long)
                    model = run["model"] if run["model"] else "N/A"
                    if model and len(model) > 15:
                        if "-" in model:
                            parts = model.split("-")
                            if len(parts) > 3:
                                model = "-".join(parts[:-1])
                        else:
                            model = model[:12] + "..."
                    
                    # Determine run status
                    network_file = os.path.join(run["path"], "network.json")
                    status = "✅" if os.path.exists(network_file) else "❌"
                    
                    # Add row to table with consistent width formatting
                    table_data.append([
                        f"{i + 1:2d}",
                        f"{date_str:16}",
                        f"{spec_name:12}",
                        f"{model:12}",
                        f"{task:30}",
                        f"{str(run['num_nodes'] or 'N/A'):5}",
                        f"{str(run['num_edges'] or 'N/A'):5}",
                        f"{duration_str:6}",
                        f"{status:4}"
                    ])
                
                # Display table with grid format and proper column alignment
                print('\n')
                print(tabulate(table_data, headers=headers, tablefmt="simple", 
                             colalign=("right","left","left","left","left","right","right","right","center")))
                
                input("\nPress Enter to continue...")
            
            elif choice == '5':  # Run verification on selection
                if not filtered_runs:
                    print("\nNo runs selected!")
                    input("Press Enter to continue...")
                    continue
                
                print(f"\nRunning verification on {len(filtered_runs)} runs...")
                for run in filtered_runs:
                    spec_match = find_matching_spec(run["spec_name"], specs_dir) if run["spec_name"] else None
                    if spec_match:
                        print(f"\nVerifying {run['name']}...")
                        try:
                            results = verify_network_trajectory(run['path'], spec_match)
                            print(f"Result: {'✅ Passed' if results['all_passed'] else '❌ Failed'}")
                            
                            # Show detailed results
                            if not results['all_passed']:
                                print("\nFailed checks:")
                                for check in results['subpath_checks']:
                                    if not check['passed']:
                                        print(f"  ❌ {check['details']}")
                                for check in results['edge_checks']:
                                    if not check['passed']:
                                        print(f"  ❌ {check['details']}")
                        except Exception as e:
                            print(f"Error: {e}")
                    else:
                        print(f"No matching spec found for {run['name']}")
                input("\nPress Enter to continue...")
            
            elif choice == '6':  # Run analysis on selection
                if not filtered_runs:
                    print("\nNo runs selected!")
                    input("Press Enter to continue...")
                    continue
                
                print(f"\nRunning analysis on {len(filtered_runs)} runs...")
                for run in filtered_runs:
                    print(f"\nAnalyzing {run['name']}...")
                    try:
                        analysis = analyze_network_logs(run['path'])
                        print(f"Nodes: {analysis['num_nodes']}, Edges: {analysis['num_edges']}")
                        print("Top central nodes:")
                        for node, centrality in analysis['central_nodes'][:3]:
                            print(f"  {node}: {centrality:.4f}")
                        
                        # Show node degrees
                        print("\nNode degrees:")
                        sorted_degrees = sorted(analysis['node_degrees'].items(), 
                                             key=lambda x: x[1], reverse=True)[:5]
                        for node, degree in sorted_degrees:
                            print(f"  {node}: {degree}")
                    except Exception as e:
                        print(f"Error: {e}")
                input("\nPress Enter to continue...")
            
            elif choice == '7':
                break
            
            else:
                print("\nInvalid choice")
                input("Press Enter to continue...")
    
    while True:
        if selected_run is None:
            # Display list of runs
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("\n" + "=" * 80)
            print(" " * 25 + "AGENT NETWORK ANALYSIS TOOL")
            print("=" * 80)
            
            filter_status = "Showing complete runs only" if only_complete else "Showing all runs (including incomplete)"
            print(f"\n{filter_status} - {len(runs)} runs found")
            
            total_pages = (len(runs) + page_size - 1) // page_size
            start_idx = page * page_size
            end_idx = min(start_idx + page_size, len(runs))
            
            # Create table data
            headers = ["#", "Date", "Spec", "Model", "Task", "Nodes", "Edges", "Duration", "Status"]
            table_data = []
            
            for i, run in enumerate(runs[start_idx:end_idx]):
                # Format timestamp
                date_str = run["timestamp"].strftime("%Y-%m-%d %H:%M") if run["timestamp"] else "Unknown"
                
                # Format duration
                if run["duration"] is not None:
                    if run["duration"] < 60:
                        duration_str = f"{run['duration']:.1f}s"
                    elif run["duration"] < 3600:
                        duration_str = f"{run['duration']/60:.1f}m"
                    else:
                        duration_str = f"{run['duration']/3600:.1f}h"
                else:
                    duration_str = "N/A"
                
                # Format task (truncate if too long)
                task = run["task"] if run["task"] else "N/A"
                if task and len(task) > 30:
                    task = task[:27] + "..."
                
                # Format spec name (truncate if too long)
                spec_name = run["spec_name"] or "Unknown"
                if len(spec_name) > 15:
                    spec_name = spec_name[:12] + "..."
                
                # Format model (truncate if too long)
                model = run["model"] if run["model"] else "N/A"
                if model and len(model) > 15:
                    # Extract the model name from the full string if it's too long
                    if "-" in model:
                        # Try to get just the base model name (e.g., "claude-3-opus" from "claude-3-opus-20240229")
                        parts = model.split("-")
                        if len(parts) > 3:
                            # Keep only the first parts that define the model type
                            model = "-".join(parts[:-1])
                    else:
                        model = model[:12] + "..."
                
                # Determine run status
                network_file = os.path.join(run["path"], "network.json")
                status = "✅" if os.path.exists(network_file) else "❌"
                
                # Add row to table with consistent width formatting
                table_data.append([
                    f"{start_idx + i + 1:2d}",  # Reduced width for #
                    f"{date_str:16}",
                    f"{spec_name:12}",  # Reduced width for spec
                    f"{model:12}",      # Reduced width for model
                    f"{task:30}",
                    f"{str(run['num_nodes'] or 'N/A'):5}",  # Reduced width for nodes
                    f"{str(run['num_edges'] or 'N/A'):5}",  # Reduced width for edges
                    f"{duration_str:6}",  # Reduced width for duration
                    f"{status:4}"        # Centered status with fixed width
                ])
            
            # Display table with grid format and proper column alignment
            print('\n')
            print(tabulate(table_data, headers=headers, tablefmt="simple", 
                         colalign=("right","left","left","left","left","right","right","right","center")))
            
            print('\n')
            
            # Display pagination info
            if total_pages > 1:
                print(f"\nPage {page+1}/{total_pages} - Showing runs {start_idx+1}-{end_idx} of {len(runs)}")
            
            # Display menu
            print("\nOptions:")
            print("  [number] - Select run by number")
            if page > 0:
                print("  p - Previous page")
            if page < total_pages - 1:
                print("  n - Next page")
            print("  f - Toggle filter: " + ("Show all runs" if only_complete else "Show complete runs only"))
            print("  b - Batch operations")
            print("  r - Refresh run list")
            print("  q - Quit")
            
            choice = input("\nEnter your choice: ").strip().lower()
            
            if choice == 'q':
                break
            elif choice == 'p' and page > 0:
                page -= 1
            elif choice == 'n' and page < total_pages - 1:
                page += 1
            elif choice == 'f':
                # Toggle filter between complete and all runs
                only_complete = not only_complete
                runs = list_runs(log_dir, only_complete=only_complete)
                page = 0
            elif choice == 'b':
                batch_operations()
            elif choice == 'r':
                runs = list_runs(log_dir, only_complete=only_complete)
                page = 0
            elif choice.isdigit():
                run_idx = int(choice) - 1
                if 0 <= run_idx < len(runs):
                    selected_run = runs[run_idx]
                else:
                    print(f"Invalid run number: {choice}")
                    input("Press Enter to continue...")
            else:
                print(f"Invalid choice: {choice}")
                input("Press Enter to continue...")
        
        else:
            # Display selected run details and options
            os.system('cls' if os.name == 'nt' else 'clear')
            
            run = selected_run
            print("\n" + "=" * 80)
            print(f" RUN: {run['name']}")
            print("=" * 80)
            
            # Basic info
            print("\nBasic Information:")
            print(f"  Date: {run['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if run['timestamp'] else 'Unknown'}")
            print(f"  Spec: {run['spec_name'] or 'Unknown'}")
            print(f"  Model: {run['model'] or 'Unknown'}")
            print(f"  Task: {run['task'] or 'Unknown'}")
            print(f"  Nodes: {run['num_nodes'] or 'N/A'}")
            print(f"  Edges: {run['num_edges'] or 'N/A'}")
            if run["duration"] is not None:
                print(f"  Duration: {run['duration']:.2f} seconds")
            
            # Check for spec match
            spec_match = None
            if run["spec_name"]:
                spec_match = find_matching_spec(run["spec_name"], specs_dir)
            
            # Display menu
            print("\nOptions:")
            print("  a - Analyze network")
            print("  v - Visualize network")
            if spec_match:
                print("  s - View spec details")
                print("  t - Verify trajectory against spec")
            print("  b - Back to run list")
            print("  q - Quit")
            
            choice = input("\nEnter your choice: ").strip().lower()
            
            if choice == 'q':
                break
            elif choice == 'b':
                selected_run = None
            elif choice == 'a':
                # Analyze network
                print(f"\nAnalyzing network from {run['path']}...")
                analysis = analyze_network_logs(run['path'])
                
                print("\n=== Network Analysis ===")
                print(f"Number of nodes: {analysis['num_nodes']}")
                print(f"Number of edges: {analysis['num_edges']}")
                print("\nNode degrees:")
                for node, degree in analysis['node_degrees'].items():
                    print(f"  {node}: {degree}")
                
                print("\nCentral nodes:")
                for node, centrality in analysis['central_nodes']:
                    print(f"  {node}: {centrality:.4f}")
                
                input("\nPress Enter to continue...")
            
            elif choice == 'v':
                # Visualize network
                print(f"\nVisualizing network from {run['path']}...")
                input("Press Enter to start visualization...")
                visualize_network_ascii(run['path'])
            
            elif choice == 's' and spec_match:
                # View spec details
                try:
                    with open(spec_match, 'r') as f:
                        spec_data = json.load(f)
                    
                    print(f"\n=== Spec: {os.path.basename(spec_match)} ===")
                    print(f"Task: {spec_data.get('task', 'N/A')}")
                    print(f"Number of agents: {len(spec_data.get('agents', []))}")
                    print(f"Number of connections: {len(spec_data.get('connections', []))}")
                    
                    print("\nAgents:")
                    for agent in spec_data.get('agents', []):
                        print(f"  - {agent.get('name', 'Unnamed')} ({agent.get('role', 'No role')})")
                    
                    input("\nPress Enter to continue...")
                except Exception as e:
                    print(f"Error reading spec file: {e}")
                    input("\nPress Enter to continue...")
            
            elif choice == 't' and spec_match:
                # Verify trajectory against spec
                print(f"\nVerifying network trajectory from {run['path']} against spec {spec_match}")
                try:
                    results = verify_network_trajectory(run['path'], spec_match)
                    
                    print("\n=== Verification Results ===")
                    print(f"Overall Result: {'Passed' if results['all_passed'] else 'Failed'}")
                    
                    print("\nSubpath Checks:")
                    for i, check in enumerate(results['subpath_checks']):
                        status = "✅" if check['passed'] else "❌"
                        print(f"{status} {check['details']}")
                        print(f"    Path: {' → '.join(check['subpath'])}")
                    
                    print("\nEdge Checks:")
                    for i, check in enumerate(results['edge_checks']):
                        status = "✅" if check['passed'] else "❌"
                        check_info = check['check']
                        print(f"{status} From: {check_info['from']} → To: {check_info['to']}, Type: {check_info.get('type', 'any')}")
                        print(f"    Details: {check['details']}")
                    
                    input("\nPress Enter to continue...")
                except Exception as e:
                    print(f"Error verifying trajectory: {e}")
                    input("\nPress Enter to continue...")
            
            else:
                print(f"Invalid choice: {choice}")
                input("Press Enter to continue...")

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Network Graph Analysis and Visualization Tool")
    
    # Adjust arguments so all are optional
    parser.add_argument("--log-file", "-l", help="Path to the network log JSON file or run directory")
    parser.add_argument("--log-dir", "-d", default="../../logs", help="Path to the logs directory")
    parser.add_argument("--verify", "-v", help="Path to the specification JSON file for verification")
    parser.add_argument("--spec-dir", "-s", help="Path to the directory containing network spec files")
    parser.add_argument("--visualize", "-vis", action="store_true", help="Visualize the network graph")
    parser.add_argument("--analyze", "-a", action="store_true", help="Analyze the network graph")
    parser.add_argument("--interactive", "-i", action="store_true", help="Use interactive browser (default if no other options specified)")
    parser.add_argument("--all-runs", action="store_true", help="Include incomplete runs in the browser")
    
    args = parser.parse_args()
    
    # If a specific log file is provided, use it for analysis, verification, or visualization
    if args.log_file:
        if args.verify:
            print(f"Verifying network trajectory from {args.log_file} against spec {args.verify}")
            results = verify_network_trajectory(args.log_file, args.verify)
            
            # Print results
            print("\n=== Verification Results ===")
            print(f"Overall Result: {'Passed' if results['all_passed'] else 'Failed'}")
            
            print("\nSubpath Checks:")
            for i, check in enumerate(results['subpath_checks']):
                status = "✅" if check['passed'] else "❌"
                print(f"{status} {check['details']}")
                print(f"    Path: {' → '.join(check['subpath'])}")
            
            print("\nEdge Checks:")
            for i, check in enumerate(results['edge_checks']):
                status = "✅" if check['passed'] else "❌"
                check_info = check['check']
                print(f"{status} From: {check_info['from']} → To: {check_info['to']}, Type: {check_info.get('type', 'any')}")
                print(f"    Details: {check['details']}")
        
        if args.analyze or (not args.verify and not args.visualize and not args.interactive):
            print(f"Analyzing network from {args.log_file}")
            analysis = analyze_network_logs(args.log_file)
            
            # Print analysis results
            print("\n=== Network Analysis ===")
            print(f"Number of nodes: {analysis['num_nodes']}")
            print(f"Number of edges: {analysis['num_edges']}")
            print("\nNode degrees:")
            for node, degree in analysis['node_degrees'].items():
                print(f"  {node}: {degree}")
            
            print("\nCentral nodes:")
            for node, centrality in analysis['central_nodes']:
                print(f"  {node}: {centrality:.4f}")
        
        if args.visualize or (not args.verify and not args.analyze and not args.interactive):
            print(f"Visualizing network from {args.log_file}")
            visualize_network_ascii(args.log_file)
    
    # If no specific log file or we're in interactive mode, use the browser
    elif args.interactive or not (args.verify or args.analyze or args.visualize):
        # If interactive browser is manually requested or no other options are specified
        # Initialize with complete runs only unless --all-runs is specified
        only_complete = not args.all_runs
        if args.all_runs:
            print("Including all runs (complete and incomplete)")
        interactive_run_browser(args.log_dir, args.spec_dir, only_complete=only_complete) 