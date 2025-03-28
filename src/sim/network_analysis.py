import json
import logging
import os
import datetime
import networkx as nx
from collections import defaultdict

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