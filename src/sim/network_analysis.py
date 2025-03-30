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
import re
from typing import Dict, List, Optional, Tuple, Any

class NetworkLogger:
    def __init__(self, log_dir="logs", run_config=None, spec_name=None):
        # Create a timestamp for the run
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a unique run directory using timestamp, spec name, and model name
        run_name_parts = [self.timestamp]
        if spec_name:
            # Extract basename without extension if spec_name is a path
            if '/' in spec_name:
                spec_name_base = os.path.basename(spec_name)
            else:
                spec_name_base = spec_name
            spec_name_base = os.path.splitext(spec_name_base)[0]
            run_name_parts.append(spec_name_base)
        
        # Extract model name from config if available
        if run_config and 'model' in run_config:
            model_name = str(run_config['model']) # Ensure it's a string
            # Basic sanitization for directory name (replace non-alphanumeric except _-)
            model_name = re.sub(r'[^\w\\-]+', '_', model_name) 
            run_name_parts.append(model_name)

        # Join parts to create the run name
        run_name = "_".join(run_name_parts)
        
        # Create full run directory path
        self.run_dir = os.path.join(log_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Set paths for log files
        self.log_file = os.path.join(self.run_dir, f"agent_network.log")
        self.network_file = os.path.join(self.run_dir, f"network.json")

        self.spec_name_file = os.path.join(self.run_dir, "spec_path.txt")
        
        # Save run configuration if provided
        if run_config:
            self.save_config(run_config)

        # Save spec name if provided
        if spec_name:
            with open(self.spec_name_file, "w") as f:
                f.write(spec_name)
            
        # Copy the spec file if provided and exists
        # Use the original spec_name path here, not the potentially modified base name
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
                    # Ensure the key exists in log arguments and the value matches exactly
                    if key not in log['arguments'] or log['arguments'][key] != expected_value:
                        args_match = False
                        break
                
                # Also ensure no extra arguments exist in the log unless expected_args is empty
                if expected_args and len(log['arguments']) != len(expected_args):
                     # If expected_args is defined, we expect an exact match of keys
                     pass # Allow extra args as per user request: "additional fields or subfields do NOT prevent a match"

                if args_match:
                    return True, f"Found matching tool call from {from_node} to {to_node} with expected arguments"
            
            # Check content for other message types
            elif 'content' in log and 'content' in check and log['content'] == check.get('content'):
                 # Check if content field exists in the check before comparing
                return True, f"Found matching message from {from_node} to {to_node} with expected content"

        # If loop finishes without returning True, no exact match was found
        return False, f"No exact match found for the specified criteria"

    # For contains check
    elif check_type == 'assertContains':
        expected_content = check.get('content') # Get expected content once
        for log in edge_logs:
            if msg_type == 'tool_call' and 'arguments' in log:
                args_contain = True
                for key, expected_value_part in expected_args.items():
                    if key not in log['arguments'] or str(expected_value_part) not in str(log['arguments'][key]):
                        args_contain = False
                        break
                
                if args_contain:
                    return True, f"Found tool call containing expected arguments pattern"
            
            elif 'content' in log and expected_content is not None and expected_content in log['content']:
                return True, f"Found message containing expected content pattern"
    
    # For exists check (just checks if the edge exists with the type filter)
    elif check_type == 'exists':
        # We already filtered by type, so if edge_logs is not empty, it exists.
        if edge_logs:
            return True, f"Edge {from_node} → {to_node} with type {msg_type or 'any'} exists"
        else:
             # This case should technically be caught earlier if no logs are found at all
             # but adding it here for clarity if the edge exists but not with the specified type.
             return False, f"Edge {from_node} → {to_node} exists, but no logs match type {msg_type}"

    # For assertNotEqual check
    elif check_type == 'assertNotEqual':
        match_found = False
        violating_log_step = None
        expected_content = check.get('content') # Get expected content once

        for log in edge_logs:
            log_matches_criteria = True # Assume this log matches unless proven otherwise

            # Check arguments if specified
            if expected_args:
                if not (log.get('type') == 'tool_call' and 'arguments' in log):
                    log_matches_criteria = False # Log type/structure doesn't match expectation
                else:
                    log_args = log['arguments']
                    for key, expected_value in expected_args.items():
                        if key not in log_args or log_args[key] != expected_value:
                            log_matches_criteria = False # Argument mismatch
                            break
                    # We don't check for extra arguments, only that the specified ones match

            # Check content if specified and arguments (if checked) matched
            if log_matches_criteria and expected_content is not None:
                if 'content' not in log or log['content'] != expected_content:
                    log_matches_criteria = False # Content mismatch

            # If this log matches all specified criteria, then assertNotEqual fails
            if log_matches_criteria:
                match_found = True
                violating_log_step = log.get('step')
                break # Found one violating log, no need to check further

        if match_found:
            # A log matching the criteria was found, violating the assertNotEqual condition
            details = f"Found matching log entry (step {violating_log_step}) violating assertNotEqual for {from_node} → {to_node}"
            if msg_type: details += f" with type {msg_type}"
            if expected_args: details += f" and args {expected_args}"
            if expected_content is not None: details += f" and content '{expected_content}'"
            return False, details
        else:
            # No log matching the exact criteria was found, assertNotEqual condition satisfied
            details = f"No log entries found matching the specified criteria for {from_node} → {to_node}"
            if msg_type: details += f" with type {msg_type}"
            if expected_args: details += f" and args {expected_args}"
            if expected_content is not None: details += f" and content '{expected_content}'"
            details += ". assertNotEqual satisfied."
            return True, details

    # Default case if check_type is unknown
    return False, f"Unknown check_type: {check_type}"

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
        "spec_name": None, # Default to None
        "model_from_dir": None, # Track model parsed from directory name
        "num_nodes": None,
        "num_edges": None,
        "task": None,
        "has_result": False,
        "duration": None,
        "model": None # Final model name (prefer config, fallback to dir)
    }
    
    # --- Start: Added logic to read spec_path.txt ---
    spec_path_file = os.path.join(run_dir, "spec_path.txt")
    if os.path.exists(spec_path_file):
        try:
            with open(spec_path_file, 'r') as f:
                original_spec_path = f.readline().strip()
                if original_spec_path:
                    spec_filename = os.path.basename(original_spec_path)
                    spec_name_base, _ = os.path.splitext(spec_filename)
                    info["spec_name"] = spec_name_base # Set spec_name from file content
        except Exception as e:
            # Log or print warning if needed
            # print(f"Warning: Could not read or parse spec_path.txt in {run_dir}: {e}")
            pass # Continue even if file reading fails
    # --- End: Added logic ---

    # Extract timestamp, spec name (if not already found), and model name from directory name using regex
    name = info["name"]
    
    # Regex to capture TIMESTAMP_SPEC_MODEL or TIMESTAMP_SPEC
    # Group 1: Timestamp (YYYYMMDD_HHMMSS)
    # Group 2: Spec Name (must contain at least one char, non-underscore)
    # Group 3: Model Name (optional, captures everything after the last underscore)
    pattern_datetime = re.compile(r'^(\d{8}_\d{6})_([^_]+?)(?:_(.+))?$')
    match = pattern_datetime.match(name)
    
    if match:
        timestamp_str, spec_name_part, model_name_part = match.groups()
        try:
            info["timestamp"] = datetime.datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            # Only set spec_name from dir if not already set from spec_path.txt
            if info["spec_name"] is None:
                 info["spec_name"] = spec_name_part
            info["model_from_dir"] = model_name_part # Can be None if not present
        except ValueError:
            pass # Ignore if timestamp parsing fails
    else:
        # Fallback: Regex for DATE_SPEC_MODEL or DATE_SPEC
        # Group 1: Timestamp (YYYYMMDD)
        # Group 2: Spec Name
        # Group 3: Model Name (optional)
        pattern_date = re.compile(r'^(\d{8})_([^_]+?)(?:_(.+))?$')
        match = pattern_date.match(name)
        if match:
            date_str, spec_name_part, model_name_part = match.groups()
            try:
                info["timestamp"] = datetime.datetime.strptime(date_str, "%Y%m%d")
                # Only set spec_name from dir if not already set from spec_path.txt
                if info["spec_name"] is None:
                    info["spec_name"] = spec_name_part
                info["model_from_dir"] = model_name_part
            except ValueError:
                 pass # Ignore if date parsing fails

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
        except Exception as e: # Broad exception catch
             # Log or print warning if needed
             # print(f"Warning: Could not read or parse config file {config_file}: {e}")
             pass
    
    # If model wasn't found in config, use the one parsed from the directory name as fallback
    if info["model"] is None and info["model_from_dir"] is not None:
         info["model"] = info["model_from_dir"]

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
    
    # Clean up temporary field
    if "model_from_dir" in info:
        del info["model_from_dir"]
        
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

def batch_verify_runs(
    log_dir: str,
    specs_dir: str,
    filters: Optional[Dict[str, Any]] = None,
    only_complete: bool = True
) -> List[Dict[str, Any]]:
    """
    Run network trajectory verification for a batch of runs filtered by specified criteria.

    Args:
        log_dir: Path to the main directory containing run logs.
        specs_dir: Path to the directory containing specification files.
        filters: Dictionary specifying filters (e.g., {"model": "model_name", "spec_name": "spec_name"}).
        only_complete: If True, only process runs that have a network.json file.

    Returns:
        List of dictionaries, each containing run info, spec path, verification result, and status.
    """
    runs_info = list_runs(log_dir, only_complete=only_complete)
    filtered_runs = []

    if filters:
        for run in runs_info:
            match = True
            for key, value in filters.items():
                if key not in run or run[key] != value:
                    # Basic exact match, could be extended for pattern matching etc.
                    match = False
                    break
            if match:
                filtered_runs.append(run)
    else:
        # If no filters, process all listed runs
        filtered_runs = runs_info

    batch_results = []
    for run in filtered_runs:
        result_entry = {
            "run_info": run,
            "spec_path": None,
            "verification_result": None,
            "status": "Unknown"
        }

        spec_match = None
        if run.get("spec_name"):
            spec_match = find_matching_spec(run["spec_name"], specs_dir)
        
        if not spec_match:
            result_entry["status"] = "No Matching Spec Found"
            batch_results.append(result_entry)
            continue

        result_entry["spec_path"] = spec_match
        
        # Ensure the run path exists and network.json is present for verification
        network_file = os.path.join(run['path'], "network.json")
        if not os.path.exists(network_file):
             result_entry["status"] = "Network Log Missing"
             batch_results.append(result_entry)
             continue

        try:
            verification_output = verify_network_trajectory(run['path'], spec_match)
            result_entry["verification_result"] = verification_output
            if verification_output.get('all_passed', False):
                result_entry["status"] = "Verified - Passed"
            else:
                result_entry["status"] = "Verified - Failed"
        except FileNotFoundError as e:
             result_entry["status"] = f"Verification Error: {e}"
        except Exception as e:
            result_entry["status"] = f"Verification Error: Unexpected error - {e}"
            # Optionally log the full error traceback here
        
        batch_results.append(result_entry)

    return batch_results

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

def compute_completion_rate(verification_results: Dict[str, Any]) -> float:
    """
    Compute the completion rate from verification results.

    This is defined as the percentage of subpaths in the specification that were
    found in the run's trajectory + the number of edge checks that passed.

    More concretely, for a given trajectory:

    rtn = (num_subpaths_passed + num_edge_checks_passed) / (num_subpaths + num_edge_checks)

    Args:
        verification_results: The dictionary returned by verify_network_trajectory.

    Returns:
        float: The completion rate (0.0 to 1.0).
    """
    subpath_checks = verification_results.get('subpath_checks', [])
    edge_checks = verification_results.get('edge_checks', [])
    
    if not subpath_checks and not edge_checks:
        return 1.0  # No checks to verify, so 100% complete by definition
        
    total_checks = len(subpath_checks) + len(edge_checks)
    passed_subpaths = sum(1 for check in subpath_checks if check['passed'])
    passed_edges = sum(1 for check in edge_checks if check['passed'])
    
    return (passed_subpaths + passed_edges) / total_checks

def compute_veracity_rate(verification_results: Dict[str, Any]) -> float:
    """
    Compute the veracity rate from verification results.

    This is just a 0-1 metric that is 1 if all subpath checks and edge checks passed.
    Else 0.0.

    Args:
        verification_results: The dictionary returned by verify_network_trajectory.

    Returns:
        float: The veracity rate (0.0 to 1.0).
    """
    subpath_checks = verification_results.get('subpath_checks', [])
    edge_checks = verification_results.get('edge_checks', [])
    
    if not subpath_checks and not edge_checks:
        return 1.0  # No checks to verify, so 100% veracious by definition
        
    total_checks = len(subpath_checks) + len(edge_checks)
    passed_subpaths = sum(1 for check in subpath_checks if check['passed'])
    passed_edges = sum(1 for check in edge_checks if check['passed'])
    
    # Only return 1.0 if ALL checks passed, otherwise 0.0
    return 1.0 if (passed_subpaths + passed_edges) == total_checks else 0.0

def compute_efficiency(log_data: Dict[str, Any], verification_results: Dict[str, Any]) -> float:
    """
    Compute the efficiency of a given run based on verification results.

    This is defined as the fraction of total steps (edges traversed in sequence_log)
    that contributed to a *passed* specified subpath or edge check. A step is
    considered contributing if it forms part of a successfully verified subpath
    or corresponds to a successfully verified edge check.

    Args:
        log_data: The network log JSON data (output of loading network.json).
        verification_results: The dictionary returned by verify_network_trajectory.

    Returns:
        float: The efficiency score (0.0 to 1.0). Returns 0.0 if there are no steps.
    """
    sequence_log = log_data.get('sequence_log', [])
    total_steps = len(sequence_log)

    if total_steps == 0:
        return 0.0 # No steps taken, efficiency is 0

    contributing_steps = set() # Use a set to avoid double-counting steps

    # 1. Identify steps contributing to passed subpaths
    #    This requires re-checking subpaths and mapping found subsequences back to steps.
    #    For simplicity, we'll approximate this by marking steps involved in *any* edge
    #    that is part of *any* specified subpath, if that subpath check passed overall.
    #    A more precise implementation would track the specific indices used in the subsequence match.
    actual_path = []
    step_edge_map = {} # Map step index to (from_node, to_node)
    for i, entry in enumerate(sequence_log):
        from_node = entry.get('from')
        to_node = entry.get('to')
        step_edge_map[i] = (from_node, to_node)
        # Build actual path for subpath checking (simplified version used in verification)
        if not actual_path or actual_path[-1] != from_node:
            actual_path.append(from_node)
        if not actual_path or actual_path[-1] != to_node:
             actual_path.append(to_node)

    passed_subpath_checks = [check for check in verification_results.get('subpath_checks', []) if check['passed']]
    for check in passed_subpath_checks:
        subpath = check['subpath']
        # Simplified: Mark all steps whose edge exists within the passed subpath transitions
        subpath_edges = set(zip(subpath[:-1], subpath[1:]))
        for step_index, (from_node, to_node) in step_edge_map.items():
            if (from_node, to_node) in subpath_edges:
                contributing_steps.add(step_index) # Add step index

    # 2. Identify steps contributing to passed edge checks
    passed_edge_checks_results = [res for res in verification_results.get('edge_checks', []) if res['passed']]
    edge_logs_by_key = log_data.get('edge_logs', {})

    for result in passed_edge_checks_results:
        check_details = result['check']
        from_node = check_details.get('from')
        to_node = check_details.get('to')
        edge_key = f"{from_node}→{to_node}"

        # Find the steps associated with this edge that match the check criteria
        # This requires revisiting the logic within verify_edge_check, which is complex.
        # Approximation: Add *all* steps associated with the edge if the check passed.
        # This overestimates efficiency if only *some* logs on an edge satisfied the check.
        logs_for_edge = edge_logs_by_key.get(edge_key, [])
        for log_entry in logs_for_edge:
            # We need the step index. Assuming 'step' in log_entry corresponds to sequence_log index + 1
            step_index = log_entry.get('step')
            if step_index is not None:
                 # Ensure the log entry *actually* matches the passed check criteria (simplified check here)
                 # We'll assume if the overall check passed, any log entry for that edge contributes
                 # Correct implementation needs to re-evaluate the check per log entry.
                 contributing_steps.add(step_index - 1) # Step is 1-based, index is 0-based

    # Calculate efficiency
    num_contributing_steps = len(contributing_steps)
    efficiency = num_contributing_steps / total_steps if total_steps > 0 else 0.0

    return efficiency

def compute_eval(run_name: str, log_dir: str = "logs", specs_dir: str = None) -> Dict[str, Any]:
    """
    Computes completion rate, veracity rate, and efficiency for a given run.

    Args:
        run_name: The name of the run directory (e.g., '20231027_120000_spec_model').
        log_dir: The base directory where run logs are stored.
        specs_dir: The directory where spec files are stored. If None, attempts default.

    Returns:
        A dictionary containing the status and computed metrics:
        {
            "status": "success" | "error",
            "error_message": str | None,
            "completion_rate": float | None,
            "veracity_rate": float | None,
            "efficiency": float | None
        }
    """
    results = {
        "status": "error",
        "error_message": None,
        "completion_rate": None,
        "veracity_rate": None,
        "efficiency": None
    }

    run_dir = os.path.join(log_dir, run_name)
    if not os.path.isdir(run_dir):
        results["error_message"] = f"Run directory not found: {run_dir}"
        return results

    # Get run info to find the associated spec name
    try:
        run_info = get_run_info(run_dir)
        spec_name = run_info.get("spec_name")
        if not spec_name:
            results["error_message"] = f"Could not determine spec name for run: {run_name}"
            return results
    except Exception as e:
        results["error_message"] = f"Error getting run info for {run_name}: {e}"
        return results

    # Find the matching spec file
    try:
        spec_path = find_matching_spec(spec_name, specs_dir)
        if not spec_path or not os.path.exists(spec_path):
            results["error_message"] = f"Matching spec file not found for spec name: {spec_name}"
            return results
    except Exception as e:
        results["error_message"] = f"Error finding spec file for {spec_name}: {e}"
        return results

    # Define network log file path
    network_file = os.path.join(run_dir, "network.json")
    if not os.path.exists(network_file):
        results["error_message"] = f"Network log file not found: {network_file}"
        return results

    # Load log data
    try:
        with open(network_file, 'r') as f:
            log_data = json.load(f)
    except json.JSONDecodeError as e:
        results["error_message"] = f"Error decoding JSON from {network_file}: {e}"
        return results
    except Exception as e:
        results["error_message"] = f"Error reading network log {network_file}: {e}"
        return results

    # Perform verification
    try:
        verification_results = verify_network_trajectory(run_dir, spec_path)
    except Exception as e:
        results["error_message"] = f"Error during verification for run {run_name}: {e}"
        return results

    # Compute metrics
    try:
        results["completion_rate"] = compute_completion_rate(verification_results)
        results["veracity_rate"] = compute_veracity_rate(verification_results)
        results["efficiency"] = compute_efficiency(log_data, verification_results)
        results["status"] = "success"
    except Exception as e:
        # If computation fails after successful verification
        results["error_message"] = f"Error computing metrics for run {run_name}: {e}"
        # Keep status as error, metrics might be partially computed or None

    return results

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Network Graph Analysis and Verification Tool")
    
    parser.add_argument("--log-file", "-l", help="Path to the network log JSON file or run directory")
    parser.add_argument("--log-dir", "-d", default="logs", help="Path to the logs directory")
    parser.add_argument("--verify", "-v", help="Path to the specification JSON file for verification")
    parser.add_argument("--spec-dir", "-s", help="Path to the directory containing network spec files")
    parser.add_argument("--analyze", "-a", action="store_true", help="Analyze the network graph")
    
    args = parser.parse_args()
    
    # --- Keep the logic for direct operations ---
    operation_performed = False
    if args.log_file:
        if args.verify:
            operation_performed = True
            print(f"Verifying network trajectory from {args.log_file} against spec {args.verify}")
            try:
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
            except Exception as e:
                print(f"Error during verification: {e}")

        if args.analyze:
            operation_performed = True
            print(f"Analyzing network from {args.log_file}")
            try:
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
            except Exception as e:
                print(f"Error during analysis: {e}")

        if not operation_performed:
             print(f"No operation (analyze, verify) specified for log file: {args.log_file}")
             print("Use --analyze or --verify <spec_file>.")

    elif not args.log_file:
         # If no log file and no interactive mode, show help
         parser.print_help()
         sys.exit(1) 