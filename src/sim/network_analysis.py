import json
import logging
import os
import datetime
import networkx as nx
from collections import defaultdict
import argparse

class NetworkLogger:
    def __init__(self, log_dir="logs"):
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Create timestamp for unique log file name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"agent_network_{timestamp}.log")
        
        # Set up logger
        self.logger = logging.getLogger("agent_network")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        
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
        
        # Generate filename
        base_filename = os.path.splitext(self.log_file)[0]
        network_file = f"{base_filename}_network.json"
        
        # Save to file
        with open(network_file, 'w') as f:
            json.dump(network_data, f, indent=2)
        
        self.logger.info(f"Network structure saved to {network_file}")
        return network_file

def analyze_network_logs(log_file):
    """Analyze the network logs to extract insights"""
    with open(log_file, 'r') as f:
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
        log_file: Path to the network log JSON file
        spec_file: Path to the specification JSON file with verification criteria
        
    Returns:
        dict: Verification results with details on which checks passed or failed
    """
    # Load network log data
    with open(log_file, 'r') as f:
        log_data = json.load(f)
    
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
        log_file: Path to the network log JSON file
    """
    with open(log_file, 'r') as f:
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

if __name__ == "__main__":
    # Example usage
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Network Graph Analysis and Visualization Tool")
    parser.add_argument("log_file", help="Path to the network log JSON file")
    parser.add_argument("--verify", "-v", help="Path to the specification JSON file for verification")
    parser.add_argument("--visualize", "-vis", action="store_true", help="Visualize the network graph")
    parser.add_argument("--analyze", "-a", action="store_true", help="Analyze the network graph")
    
    args = parser.parse_args()
    
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
    
    if args.analyze or (not args.verify and not args.visualize):
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
    
    if args.visualize or (not args.verify and not args.analyze):
        print(f"Visualizing network from {args.log_file}")
        visualize_network_ascii(args.log_file) 