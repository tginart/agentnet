import os
import sys
import json
import datetime
import argparse
import networkx as nx
from tabulate import tabulate
from typing import Dict, List, Optional, Any 
from collections import defaultdict

# Adjust path to import from src/sim
try:
    # This assumes viz.py is in the root directory
    from src.sim.network_analysis import (
        list_runs, find_matching_spec, analyze_network_logs,
        verify_network_trajectory, get_spec_info,
        batch_verify_runs, compute_eval
    )
except ImportError as e:
    # Fallback if structure is different or script is run from elsewhere
    print(f"Error: Could not import from src.sim.network_analysis: {e}")
    print("Ensure viz.py is in the project root and the 'src' directory with 'sim/network_analysis.py' exists.")
    sys.exit(1)


def eval_all_complete_runs(log_dir: str, specs_dir: str, 
    include_all_specs: Optional[bool] = False, # default is to only include specs present for all models
    model_to_evaluate: Optional[str] = None, # default is to evaluate all models
    ):
    '''
    Evaluate all complete runs in the given log directory.
    '''

    runs = list_runs(log_dir, only_complete=True)

    # breakdown runs into list of (model, spec) tuples
    runs_by_model_and_spec = defaultdict(list)
    for run in runs:
        runs_by_model_and_spec[(run['model'], run['spec_name'])].append(run)

    # get list of specs present for all models using runs_by_model_and_spec
    # Get all unique specs and models
    all_specs = {spec for _, spec in runs_by_model_and_spec.keys()}
    all_models = {model for model, _ in runs_by_model_and_spec.keys()}

    # Find specs present for all models
    specs_for_all_models = set()
    for spec in all_specs:
        present_for_all = True
        for model in all_models:
            if (model, spec) not in runs_by_model_and_spec:
                present_for_all = False
                break
        if present_for_all or include_all_specs:
            specs_for_all_models.add(spec)

    
    model_names = set()
    run_eval_results = dict()
    for run in runs:
        if run['spec_name'] in specs_for_all_models:
            model_names.add(run['model'])
            run_eval_results[run['name']] = compute_eval(run['name'], log_dir, specs_dir)
            run_eval_results[run['name']]['model'] = run['model']


    # save run_eval_results to json file
    with open('run_eval_results.json', 'w') as f:
        json.dump(run_eval_results, f, indent=4)

    print("\nEvaluating on the following environments:")
    for spec_name in specs_for_all_models:
        print(f"--- {spec_name} ---")
    print('\n')
        
    for model_name in model_names:
        # print(f"Evaluating model: {model_name}")
        if model_to_evaluate and model_name != model_to_evaluate:
            # print(f"Skipping model: {model_name}")
            continue
        avg_completion_rate = []
        avg_veracity_rate = []
        avg_efficiency = []
        for run_name, run_eval_result in run_eval_results.items():
            if run_eval_result['model'] == model_name:
                avg_completion_rate.append(run_eval_result['completion_rate'])
                avg_veracity_rate.append(run_eval_result['veracity_rate'])
                avg_efficiency.append(run_eval_result['efficiency'])


        avg_completion_rate = sum(avg_completion_rate) / len(avg_completion_rate)
        avg_veracity_rate = sum(avg_veracity_rate) / len(avg_veracity_rate)
        avg_efficiency = sum(avg_efficiency) / len(avg_efficiency)

        print(f"Model: {model_name}")
        print(f"  Completion rate: {avg_completion_rate}")
        print(f"  Veracity rate: {avg_veracity_rate}")
        print(f"  Efficiency: {avg_efficiency}")


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

def find_spec_dir(specs_dir: str = None):
    '''
    Find the specs directory.
    '''
    # Try to intelligently find specs_dir if not provided or default doesn't exist
    if specs_dir is None or not os.path.isdir(specs_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_spec_path = os.path.join(script_dir, "network_specs")
        potential_spec_dir = os.path.join(script_dir, "src", "sim", "network_specs") 
        
        if specs_dir and os.path.isdir(specs_dir):
            pass # Use the provided one if it's valid
        elif os.path.isdir(default_spec_path):
             specs_dir = default_spec_path
             print(f"Using specs directory: {specs_dir}")
        elif os.path.isdir(potential_spec_dir):
            specs_dir = potential_spec_dir
            print(f"Using specs directory: {specs_dir}")
        elif specs_dir: # Provided but not valid
             print(f"Warning: Provided specs directory '{specs_dir}' not found. Verification features might fail.")
        else: # Not provided and defaults not found
             print("Warning: Default spec directories ('network_specs', 'src/sim/network_specs') not found. Verification features might fail.")
             specs_dir = None # Ensure it's None if we can't find it
    return specs_dir

def interactive_run_browser(log_dir: str = "logs", specs_dir: str = None, only_complete: bool = True):
    """
    Interactive terminal-based browser for past runs. Imports core logic from network_analysis.
    
    Args:
        log_dir: Path to the logs directory
        specs_dir: Path to the specs directory (optional)
        only_complete: If True, only show complete runs initially (Note: This implementation now effectively ignores this and always behaves as True)
    """
    
    # Force only_complete to True internally, simplifying logic below
    only_complete = True 

    specs_dir = find_spec_dir(specs_dir)

    # Internal state for display mode
    show_all_runs_mode = False # Start in 'most recent only' mode

    # --- Helper function to get and filter runs --- 
    def _get_filtered_runs(current_log_dir: str, show_all: bool, model_filter: str = None, task_filter: str = None) -> List[Dict]:
        """Fetches and filters runs based on the mode and optional filters."""
        print("\nFetching run list...")
        all_runs_before_filtering = list_runs(current_log_dir, only_complete=True) # Fetch all first
        filtered_runs = all_runs_before_filtering
        
        # Apply model filter if specified
        if model_filter:
            filtered_runs = [run for run in filtered_runs if run.get('model') == model_filter]
            print(f"Applied model filter: {model_filter} ({len(filtered_runs)} runs)")
        
        # Apply task filter if specified (now uses exact match)
        if task_filter:
            filtered_runs = [run for run in filtered_runs if run.get('task') == task_filter]
            print(f"Applied task filter: {task_filter} ({len(filtered_runs)} runs)")
        
        if not show_all and filtered_runs:
            print("Filtering for most recent run per model-spec pair...")
            most_recent_runs_dict = {}
            filtered_runs.sort(key=lambda r: r.get('timestamp', datetime.datetime.min), reverse=True)
            
            for run in filtered_runs:
                if not run.get('timestamp'): continue
                model = run.get('model', 'Unknown')
                spec_name = run.get('spec_name', 'Unknown')
                key = (model, spec_name)
                if key not in most_recent_runs_dict:
                    most_recent_runs_dict[key] = run
            
            most_recent_filtered = sorted(list(most_recent_runs_dict.values()), key=lambda r: r.get('timestamp', datetime.datetime.min), reverse=True)
            print(f"Displaying {len(most_recent_filtered)} most recent runs.")
            return most_recent_filtered
        else:
            # Return all filtered runs if show_all is True or no runs found initially
            print(f"Displaying {len(filtered_runs)} filtered runs.")
            return filtered_runs
    # --- End Helper ---    

    # Get initial list of runs based on the default mode
    runs = _get_filtered_runs(log_dir, show_all_runs_mode, None, None)

    selected_run = None
    page = 0
    page_size = 10
    current_model_filter = None
    current_task_filter = None
    
    # --- Batch Operations Helper ---
    # (Moved from network_analysis.py)
    def batch_operations():
        """Handle batch operations on filtered runs"""
        # Note: This operates on the 'runs' list visible in the outer scope
        filtered_runs_indices = list(range(len(runs))) # Store indices to handle dynamic filtering
        current_filters_desc = [] # Track descriptions of active filters

        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("\n" + "=" * 80)
            print(" " * 25 + "BATCH OPERATIONS")
            print("=" * 80)

            # Apply filters to get current selection
            current_selection = [runs[i] for i in filtered_runs_indices]

            # Get unique values for filtering from the *original* runs list
            all_models = sorted(list(set(run["model"] for run in runs if run.get("model"))))
            all_specs = sorted(list(set(run["spec_name"] for run in runs if run.get("spec_name"))))

            # Show current filters if any
            if current_filters_desc:
                print("\nActive Filters:")
                for f in current_filters_desc:
                    print(f"  • {f}")
            else:
                print("\nNo active filters - operating on all displayed runs")
            print(f"Selected runs: {len(current_selection)}")

            print("\nFilter Options (apply sequentially):")
            print("  1. Filter by model")
            print("  2. Filter by spec name")
            print("  3. Clear all filters")

            print("\nBatch Actions:")
            print("  4. Show current selection details")
            print("  5. Run verification on current selection")
            print("  6. Run analysis on current selection")
            print("  7. Back to main run browser")

            choice = input("\nEnter your choice: ").strip()

            if choice == '1':
                if not all_models:
                    print("\nNo models found in the run data to filter by.")
                    input("Press Enter...")
                    continue
                print("\nSelect model to filter by:")
                for i, model in enumerate(all_models):
                    print(f"  {i+1}. {model}")
                model_choice = input("Enter model number: ").strip()
                if model_choice.isdigit() and 0 < int(model_choice) <= len(all_models):
                    selected_model = all_models[int(model_choice)-1]
                    # Filter the *current* selection further
                    filtered_runs_indices = [
                        i for i in filtered_runs_indices if runs[i].get("model") == selected_model
                    ]
                    current_filters_desc.append(f"Model = '{selected_model}'")
                    print(f"\nFiltered selection to {len(filtered_runs_indices)} runs.")
                else:
                    print("Invalid choice.")
                input("Press Enter...")

            elif choice == '2':
                if not all_specs:
                    print("\nNo spec names found in the run data to filter by.")
                    input("Press Enter...")
                    continue
                print("\nSelect spec name to filter by:")
                for i, spec in enumerate(all_specs):
                    print(f"  {i+1}. {spec}")
                spec_choice = input("Enter spec number: ").strip()
                if spec_choice.isdigit() and 0 < int(spec_choice) <= len(all_specs):
                    selected_spec = all_specs[int(spec_choice)-1]
                    # Filter the *current* selection further
                    filtered_runs_indices = [
                        i for i in filtered_runs_indices if runs[i].get("spec_name") == selected_spec
                    ]
                    current_filters_desc.append(f"Spec Name = '{selected_spec}'")
                    print(f"\nFiltered selection to {len(filtered_runs_indices)} runs.")
                else:
                    print("Invalid choice.")
                input("Press Enter...")

            elif choice == '3':
                filtered_runs_indices = list(range(len(runs))) # Reset to all runs
                current_filters_desc = []
                print("\nFilters cleared.")
                input("Press Enter...")

            elif choice == '4':
                print(f"\nCurrent selection: {len(current_selection)} runs")
                if current_filters_desc:
                    print("Active filters:")
                    for f in current_filters_desc: print(f"  • {f}")
                else:
                    print("No active filters.")
                
                if not current_selection:
                     print("Selection is empty.")
                     input("Press Enter...")
                     continue

                # --- Display Table Code (copied and adapted from main loop) ---
                headers = ["#", "Date", "Spec", "Model", "Task", "Nodes", "Edges", "Duration", "Status"]
                table_data = []
                for i, run_idx in enumerate(filtered_runs_indices): # Iterate through indices of selection
                    run = runs[run_idx] # Get run data using index
                    date_str = run["timestamp"].strftime("%Y-%m-%d %H:%M") if run.get("timestamp") else "Unknown"
                    duration_str = "N/A"
                    if run.get("duration") is not None:
                        dur = run["duration"]
                        if dur < 60: duration_str = f"{dur:.1f}s"
                        elif dur < 3600: duration_str = f"{dur/60:.1f}m"
                        else: duration_str = f"{dur/3600:.1f}h"
                    task = run.get("task", "N/A")
                    task_short = (task[:27] + "...") if task and len(task) > 30 else task # Keep task shortening for now

                    # Ensure spec_name is a string, default to "Unknown"
                    spec_name = str(run.get("spec_name")) if run.get("spec_name") is not None else "Unknown"
                    # Remove spec name truncation: spec_short = (spec_name[:12] + "...") if len(spec_name) > 15 else spec_name

                    # Ensure model is a string, default to "N/A"
                    model = str(run.get("model")) if run.get("model") is not None else "N/A"
                    model_short = model # Start with guaranteed string
                    if len(model) > 15:
                        if "-" in model:
                            parts = model.split("-")
                            # Shorten only if significantly long (more than 3 parts)
                            if len(parts) > 3: model_short = "-".join(parts[:-1]) + "-..."
                        else: model_short = model[:12] + "..."

                    network_file = os.path.join(run["path"], "network.json")
                    status = "✅" if os.path.exists(network_file) else "❌"

                    # Ensure final values are strings before formatting (extra safety)
                    spec_name_final = str(spec_name) # Use full spec name
                    model_short_final = str(model_short) if model_short is not None else "ERR"

                    table_data.append([
                        f"{i + 1:2d}", f"{date_str:16}", f"{spec_name_final:20}", f"{model_short_final:12}", # Increased spec width
                        f"{task_short:22}", f"{str(run.get('num_nodes', 'N/A')):5}", f"{str(run.get('num_edges', 'N/A')):5}", # Decreased task width
                        f"{duration_str:6}", f"{status:4}"
                    ])
                print('\n')
                print(tabulate(table_data, headers=headers, tablefmt="simple",
                             colalign=("right","left","left","left","left","right","right","right","center")))
                # --- End Table Code ---
                input("\nPress Enter...")

            elif choice == '5':  # Run verification on current selection
                if not current_selection:
                    print("\nNo runs selected to verify!")
                    input("Press Enter...")
                    continue
                if specs_dir is None:
                     print("\nError: Specs directory not found or specified. Cannot run verification.")
                     input("Press Enter...")
                     continue

                print(f"\nPreparing to run batch verification on {len(current_selection)} runs...")

                # Construct the filter dictionary based on the *active* filter descriptions
                # This assumes simple filters like 'Key = Value'
                batch_filters = {}
                for f_desc in current_filters_desc:
                    parts = f_desc.split('=', 1)
                    if len(parts) == 2:
                        key_str = parts[0].strip().lower()
                        val_str = parts[1].strip().strip("'") # Remove potential quotes
                        if key_str == 'model':
                            batch_filters['model'] = val_str
                        elif key_str == 'spec name':
                             batch_filters['spec_name'] = val_str
                        # Add more key mappings if needed

                print(f"Applying filters for batch verification: {batch_filters if batch_filters else 'None'}")
                print("Note: Batch verification processes runs based on filters applied to the *entire* log directory.")
                input("Press Enter to start batch verification...")

                try:
                    # Call the batch verification function from network_analysis
                    # It uses the provided filters internally on the list_runs result
                    batch_results = batch_verify_runs(
                        log_dir=log_dir,
                        specs_dir=specs_dir,
                        filters=batch_filters,
                        only_complete=True # Always check complete runs in batch mode
                    )

                    print("\n--- Batch Verification Results ---")
                    if not batch_results:
                         print("No runs matched the criteria for verification.")
                    else:
                        for result in batch_results:
                            run_name = result['run_info']['name']
                            status = result['status']
                            print(f"  Run: {run_name:<40} Status: {status}")
                            # Optionally print failure details
                            if "Failed" in status and result.get('verification_result'):
                                v_result = result['verification_result']
                                print("    Failed checks:")
                                for check in v_result.get('subpath_checks', []):
                                    if not check['passed']:
                                        print(f"      ❌ Subpath: {' → '.join(check['subpath'])} ({check['details']})")
                                for check in v_result.get('edge_checks', []):
                                    if not check['passed']:
                                        check_info = check['check']
                                        details = check.get('details', 'No details')
                                        print(f"      ❌ Edge: {check_info.get('from','?')} → {check_info.get('to','?')}, Type: {check_info.get('type','any')} ({details})")

                except FileNotFoundError as e:
                     print(f"\nError during batch verification: {e} (Check log/spec paths)")
                except Exception as e:
                    print(f"\nAn unexpected error occurred during batch verification: {e}")
                    # Consider logging traceback here for debugging

                input("\nBatch verification complete. Press Enter...")


            elif choice == '6':  # Run analysis on current selection
                if not current_selection:
                    print("\nNo runs selected to analyze!")
                    input("Press Enter...")
                    continue

                print(f"\nRunning analysis on {len(current_selection)} selected runs...")
                for run in current_selection:
                    print(f"\n--- Analyzing {run['name']} ---")
                    network_file = os.path.join(run["path"], "network.json")
                    if not os.path.exists(network_file):
                         print("  Status: Incomplete (network.json missing)")
                         continue
                    try:
                        analysis = analyze_network_logs(run['path']) # Use function from network_analysis
                        print(f"  Nodes: {analysis['num_nodes']}, Edges: {analysis['num_edges']}")
                        print("  Top central nodes (Degree Centrality):")
                        for node, centrality in analysis.get('central_nodes',[])[:3]:
                            print(f"    - {node}: {centrality:.4f}")
                        # Optional: Add more analysis details here if needed
                    except FileNotFoundError:
                         print("  Error: network.json not found (should not happen if run is complete)")
                    except Exception as e:
                        print(f"  Error during analysis: {e}")
                input("\nAnalysis complete. Press Enter...")

            elif choice == '7':
                break # Exit batch operations menu

            else:
                print("\nInvalid choice")
                input("Press Enter...")
    # --- End Batch Operations Helper ---

    # --- Main Interactive Loop ---
    while True:
        if selected_run is None:
            # Display list of runs
            os.system('cls' if os.name == 'nt' else 'clear')

            print("\n" + "=" * 80)
            print(" " * 25 + "AGENT NETWORK ANALYSIS BROWSER")
            print("=" * 80)

            if not runs:
                 print("\nNo complete runs found.") # Updated message
                 print(f"(Log directory: {os.path.abspath(log_dir)})" )
                 print(f"(Spec directory: {os.path.abspath(specs_dir) if specs_dir else 'Not specified'})" )
                 print("Default view: Showing only the most recent run per model-spec pair.") # Inform user of default
                 # Offer limited options if no runs
                 print("\nOptions:")
                 print("  r - Refresh run list")
                 print("  q - Quit")

            else:
                # Runs exist, display table and full menu
                # Update filter description based on internal state
                filter_desc = "all complete runs" if show_all_runs_mode else "most recent run per model-spec pair (default)"
                print(f"\nShowing {filter_desc} - {len(runs)} runs found")
                
                # Display active filters if any
                if current_model_filter or current_task_filter:
                    print("\nActive filters:")
                    if current_model_filter:
                        print(f"  • Model = '{current_model_filter}'")
                    if current_task_filter:
                        print(f"  • Task = '{current_task_filter}'")

                total_pages = (len(runs) + page_size - 1) // page_size
                page = max(0, min(page, total_pages - 1)) # Ensure page is valid
                start_idx = page * page_size
                end_idx = min(start_idx + page_size, len(runs))

                # --- Display Table Code (Copied and adapted) ---
                headers = ["#", "Date", "Spec", "Model", "Task", "Nodes", "Edges", "Duration", "Status"]
                table_data = []
                for i in range(start_idx, end_idx):
                     run = runs[i]
                     date_str = run["timestamp"].strftime("%Y-%m-%d %H:%M") if run.get("timestamp") else "Unknown"
                     duration_str = "N/A"
                     if run.get("duration") is not None:
                         dur = run["duration"]
                         if dur < 60: duration_str = f"{dur:.1f}s"
                         elif dur < 3600: duration_str = f"{dur/60:.1f}m"
                         else: duration_str = f"{dur/3600:.1f}h"
                     task = run.get("task", "N/A")
                     task_short = (task[:27] + "...") if task and len(task) > 30 else task # Keep task shortening for now

                     # Ensure spec_name is a string, default to "Unknown"
                     spec_name = str(run.get("spec_name")) if run.get("spec_name") is not None else "Unknown"
                     # Remove spec name truncation: spec_short = (spec_name[:12] + "...") if len(spec_name) > 15 else spec_name

                     # Ensure model is a string, default to "N/A"
                     model = str(run.get("model")) if run.get("model") is not None else "N/A"
                     model_short = model # Start with guaranteed string
                     if len(model) > 15:
                         if "-" in model:
                             parts = model.split("-")
                              # Shorten only if significantly long (more than 3 parts)
                             if len(parts) > 3: model_short = "-".join(parts[:-1]) + "-..."
                         else: model_short = model[:12] + "..."

                     network_file = os.path.join(run["path"], "network.json")
                     status = "✅" if os.path.exists(network_file) else "❌"

                     # Ensure final values are strings before formatting (extra safety)
                     spec_name_final = str(spec_name) # Use full spec name
                     model_short_final = str(model_short) if model_short is not None else "ERR"

                     table_data.append([
                         f"{i + 1:2d}", f"{date_str:16}", f"{spec_name_final:20}", f"{model_short_final:12}", # Increased spec width
                         f"{task_short:22}", f"{str(run.get('num_nodes', 'N/A')):5}", f"{str(run.get('num_edges', 'N/A')):5}", # Decreased task width
                         f"{duration_str:6}", f"{status:4}"
                     ])
                print('\n')
                print(tabulate(table_data, headers=headers, tablefmt="simple",
                             colalign=("right","left","left","left","left","right","right","right","center")))
                # --- End Table Code ---

                print('\n')

                # Display pagination info
                if total_pages > 1:
                    print(f"Page {page+1}/{total_pages} - Showing runs {start_idx+1}-{end_idx} of {len(runs)}")

                # Display menu
                print("\nOptions:")
                print("  [number] - Select run by number")
                if page > 0: print("  p - Previous page")
                if page < total_pages - 1: print("  n - Next page")
                # Add toggle option
                print("  f - Toggle view: " + ("Show most recent only" if show_all_runs_mode else "Show all runs"))
                print("  m - Filter by model")
                print("  t - Filter by task")
                print("  c - Clear all filters")
                print("  b - Batch operations")
                print("  r - Refresh run list")
                print("  q - Quit")

            choice = input("\nEnter your choice: ").strip().lower()

            if choice == 'q':
                break
            elif choice == 'p' and page > 0 and runs:
                page -= 1
            elif choice == 'n' and page < total_pages - 1 and runs:
                page += 1
            elif choice == 'b' and runs:
                batch_operations() # Enter the batch submenu
            elif choice == 'r':
                 # Refresh using the helper function and current mode
                 runs = _get_filtered_runs(log_dir, show_all_runs_mode, current_model_filter, current_task_filter)
                 page = 0
                 selected_run = None
            elif choice == 'f': # Toggle display mode
                 show_all_runs_mode = not show_all_runs_mode
                 print(f"\nSwitching view to show {'all complete runs' if show_all_runs_mode else 'most recent run per model-spec pair'}...")
                 runs = _get_filtered_runs(log_dir, show_all_runs_mode, current_model_filter, current_task_filter)
                 page = 0
                 selected_run = None
                 input("Press Enter...") # Pause to show the message
            elif choice == 'm': # Filter by model
                # Get unique models from current runs list
                all_models = sorted(list(set(run.get("model", "Unknown") for run in runs if run.get("model"))))
                if not all_models:
                    print("\nNo models found in the run data to filter by.")
                    input("Press Enter...")
                    continue
                
                print("\nSelect model to filter by (or enter 'clear' to clear model filter):")
                for i, model in enumerate(all_models):
                    print(f"  {i+1}. {model}")
                model_choice = input("Enter model number: ").strip()
                
                if model_choice.lower() == 'clear':
                    current_model_filter = None
                    print("\nCleared model filter.")
                elif model_choice.isdigit() and 0 < int(model_choice) <= len(all_models):
                    current_model_filter = all_models[int(model_choice)-1]
                    print(f"\nFiltering by model: {current_model_filter}")
                else:
                    print("\nInvalid model selection.")
                
                # Refresh with new filter
                runs = _get_filtered_runs(log_dir, show_all_runs_mode, current_model_filter, current_task_filter)
                page = 0
                input("Press Enter...")
            elif choice == 't': # Filter by task
                # Get unique tasks from current runs list
                # Use original list_runs result for task options, but filter the currently displayed 'runs'
                all_tasks = sorted(list(set(run.get("task", "Unknown") for run in _get_filtered_runs(log_dir, show_all_runs_mode, current_model_filter, None) if run.get("task")))) # Get tasks based on current model filter
                
                if not all_tasks:
                    print("\nNo tasks found in the current run selection to filter by.")
                    input("Press Enter...")
                    continue
                    
                print("\nSelect task to filter by (or enter 'clear' to clear task filter):")
                for i, task in enumerate(all_tasks):
                    # Truncate long tasks for display in the list
                    display_task = (task[:60] + "...") if len(task) > 63 else task
                    print(f"  {i+1}. {display_task}")
                task_choice = input("Enter task number: ").strip()
                
                if task_choice.lower() == 'clear':
                    current_task_filter = None
                    print("\nCleared task filter.")
                elif task_choice.isdigit() and 0 < int(task_choice) <= len(all_tasks):
                    current_task_filter = all_tasks[int(task_choice)-1]
                    print(f"\nFiltering by task: {current_task_filter}")
                else:
                    print("\nInvalid task selection.")
                
                # Refresh with new filter
                runs = _get_filtered_runs(log_dir, show_all_runs_mode, current_model_filter, current_task_filter)
                page = 0
                input("Press Enter...")
            elif choice == 'c': # Clear all filters
                if current_model_filter or current_task_filter:
                    current_model_filter = None
                    current_task_filter = None
                    print("\nCleared all filters.")
                    runs = _get_filtered_runs(log_dir, show_all_runs_mode, None, None)
                    page = 0
                else:
                    print("\nNo active filters to clear.")
                input("Press Enter...")
            elif choice.isdigit() and runs:
                try:
                    run_idx = int(choice) - 1
                    if start_idx <= run_idx < end_idx: # Check if index is on current page
                        selected_run = runs[run_idx]
                    elif 0 <= run_idx < len(runs): # Valid index but not on current page
                         # Go to the page containing the selected run
                         page = run_idx // page_size
                         selected_run = runs[run_idx]
                    else:
                        print(f"Invalid run number: {choice}. Please enter a number between {start_idx+1} and {end_idx}.")
                        input("Press Enter...")
                except ValueError:
                     print(f"Invalid input: {choice}")
                     input("Press Enter...")
            elif not runs and choice in ['p','n','b','m','t','c'] or (choice.isdigit() and not runs):
                 print("Option unavailable: No runs loaded.")
                 input("Press Enter...")
            else:
                print(f"Invalid choice: {choice}")
                input("Press Enter...")

        else:
            # --- Display selected run details and options ---
            os.system('cls' if os.name == 'nt' else 'clear')

            run = selected_run # Use the selected run object
            print("\n" + "=" * 80)
            print(f" RUN DETAILS: {run['name']}")
            print("=" * 80)

            # Basic info
            print("\nBasic Information:")
            print(f"  Path: {run.get('path', 'N/A')}")
            print(f"  Date: {run.get('timestamp', 'Unknown').strftime('%Y-%m-%d %H:%M:%S') if run.get('timestamp') else 'Unknown'}")
            print(f"  Spec Name: {run.get('spec_name', 'Unknown')}")
            print(f"  Model: {run.get('model', 'Unknown')}")
            print(f"  Task: {run.get('task', 'Unknown')}")
            print(f"  Nodes: {run.get('num_nodes', 'N/A')}")
            print(f"  Edges: {run.get('num_edges', 'N/A')}")
            if run.get("duration") is not None: print(f"  Duration: {run['duration']:.2f} seconds")
            network_file = os.path.join(run["path"], "network.json")
            print(f"  Status: {'Complete (network.json found)' if os.path.exists(network_file) else 'Incomplete (network.json missing)'}")


            # Check for spec match
            spec_match_path = None
            spec_info = None
            if run.get("spec_name") and specs_dir:
                spec_match_path = find_matching_spec(run["spec_name"], specs_dir)
                if spec_match_path:
                     try:
                         spec_info = get_spec_info(spec_match_path)
                     except Exception as e:
                          print(f"\nWarning: Could not read matching spec file {spec_match_path}: {e}")

            # Display menu
            print("\nOptions:")
            if os.path.exists(network_file):
                 print("  a - Analyze network (from network.json)")
                 print("  v - Visualize network interactively (ASCII)")
                 if spec_match_path:
                     print("  t - Verify trajectory against spec")
                 else:
                      print("  t - (Verification unavailable: No matching spec found/specified)")
            else:
                 print("  a - (Analysis unavailable: network.json missing)")
                 print("  v - (Visualization unavailable: network.json missing)")
                 print("  t - (Verification unavailable: network.json missing)")

            if spec_match_path and spec_info:
                print(f"  s - View spec details ({spec_info['name']})")
            elif run.get("spec_name"):
                 print(f"  s - (Spec details unavailable: Cannot find/read spec '{run['spec_name']}')")

            print("  l - View raw agent log (agent_network.log)")
            print("  c - View config file (config.yaml)")
            print("  j - View network data (network.json)")
            print("  x - View task/result file (task_result.yaml)")

            print("  b - Back to run list")
            print("  q - Quit")

            choice = input("\nEnter your choice: ").strip().lower()

            if choice == 'q':
                # Need to exit the outer loop as well
                selected_run = None # Deselect first
                sys.exit(0) # Exit program cleanly
            elif choice == 'b':
                selected_run = None # Go back to list view
            
            # Actions requiring network.json
            elif choice == 'a' and os.path.exists(network_file):
                print(f"\nAnalyzing network from {run['path']}...")
                try:
                    analysis = analyze_network_logs(run['path']) # Use imported function
                    print("\n--- Network Analysis ---")
                    print(f"  Number of nodes: {analysis['num_nodes']}")
                    print(f"  Number of edges: {analysis['num_edges']}")
                    print("\n  Node degrees (Top 5):")
                    sorted_degrees = sorted(analysis['node_degrees'].items(), key=lambda item: item[1], reverse=True)
                    for node, degree in sorted_degrees[:5]: print(f"    - {node}: {degree}")
                    print("\n  Central nodes (Degree Centrality - Top 5):")
                    for node, centrality in analysis.get('central_nodes',[])[:5]: print(f"    - {node}: {centrality:.4f}")
                except Exception as e:
                     print(f"Error during analysis: {e}")
                input("\nPress Enter...")

            elif choice == 'v' and os.path.exists(network_file):
                print(f"\nStarting interactive ASCII visualization for {run['path']}...")
                try:
                    visualize_network_ascii(run['path']) # Use imported function
                except Exception as e:
                     print(f"Error during visualization: {e}")
                print("\nExited visualization.")
                input("Press Enter...") # Pause after visualization exits

            elif choice == 't' and os.path.exists(network_file) and spec_match_path:
                print(f"\nVerifying trajectory from {run['path']} against spec {spec_match_path}")
                try:
                    results = verify_network_trajectory(run['path'], spec_match_path) # Use imported function
                    print("\n--- Verification Results ---")
                    print(f"  Overall Result: {'Passed' if results['all_passed'] else 'Failed'}")
                    if not results['all_passed']:
                         print("\n  Failed Checks Details:")
                         print("    Subpaths:")
                         found_failed_subpath = False
                         for i, check in enumerate(results['subpath_checks']):
                             if not check['passed']:
                                 status = "❌"
                                 print(f"      {status} {check['details']} (Path: {' → '.join(check['subpath'])})")
                                 found_failed_subpath = True
                         if not found_failed_subpath: print("      (All subpath checks passed or none defined)")

                         print("\n    Edge Checks:")
                         found_failed_edge = False
                         for i, check in enumerate(results['edge_checks']):
                             if not check['passed']:
                                 status = "❌"
                                 check_info = check['check']
                                 details = check.get('details', 'No details')
                                 print(f"      {status} From: {check_info['from']} → To: {check_info['to']}, Type: {check_info.get('type', 'any')}")
                                 print(f"          Details: {details}")
                                 found_failed_edge = True
                         if not found_failed_edge: print("      (All edge checks passed or none defined)")

                except Exception as e:
                    print(f"Error during verification: {e}")
                input("\nPress Enter...")
            
            # Actions requiring specific files
            elif choice == 's' and spec_match_path and spec_info:
                 print(f"\n--- Spec Details: {spec_info['name']} ---")
                 print(f"  Path: {spec_info['path']}")
                 print(f"  Task: {spec_info.get('task', 'N/A')}")
                 print(f"  Num Agents: {spec_info.get('num_agents', 'N/A')}")
                 print(f"  Num Connections: {spec_info.get('num_connections', 'N/A')}")
                 # Optionally load and print more details from the spec JSON
                 input("\nPress Enter...")

            elif choice == 'l': # View raw agent log
                log_path = os.path.join(run['path'], "agent_network.log")
                if os.path.exists(log_path):
                     print(f"\n--- Displaying Log: {log_path} (last 50 lines) ---")
                     try:
                         with open(log_path, 'r') as f:
                             lines = f.readlines()
                             for line in lines[-50:]: # Show tail
                                 print(line, end='')
                         print("--- End of log preview ---")
                     except Exception as e:
                         print(f"Error reading log file: {e}")
                else:
                    print(f"Log file not found: {log_path}")
                input("\nPress Enter...")

            elif choice == 'c': # View config
                 config_path = os.path.join(run['path'], "config.yaml")
                 if os.path.exists(config_path):
                      print(f"\n--- Displaying Config: {config_path} ---")
                      try:
                          with open(config_path, 'r') as f:
                              print(f.read())
                          print("--- End of config ---")
                      except Exception as e:
                          print(f"Error reading config file: {e}")
                 else:
                     print(f"Config file not found: {config_path}")
                 input("\nPress Enter...")

            elif choice == 'j': # View network json
                 net_path = os.path.join(run['path'], "network.json")
                 if os.path.exists(net_path):
                      print(f"\n--- Displaying Network JSON (partial): {net_path} ---")
                      try:
                          with open(net_path, 'r') as f:
                              # Read and potentially truncate large data
                              content = f.read(2000) # Limit initial display
                              print(content)
                              if len(content) == 2000: print("... (file truncated)")
                          print("--- End of network JSON preview ---")
                      except Exception as e:
                          print(f"Error reading network JSON file: {e}")
                 else:
                      print(f"Network JSON file not found: {net_path}")
                 input("\nPress Enter...")

            elif choice == 'x': # View task/result yaml
                 tr_path = os.path.join(run['path'], "task_result.yaml")
                 if os.path.exists(tr_path):
                      print(f"\n--- Displaying Task/Result: {tr_path} ---")
                      try:
                          with open(tr_path, 'r') as f:
                              print(f.read())
                          print("--- End of task/result ---")
                      except Exception as e:
                          print(f"Error reading task/result file: {e}")
                 else:
                      print(f"Task/Result file not found: {tr_path}")
                 input("\nPress Enter...")

            # Handle cases where option is invalid for current state
            elif choice in ['a', 'v', 't'] and not os.path.exists(network_file):
                 print("Action requires network.json, which is missing for this run.")
                 input("Press Enter...")
            elif choice == 't' and not spec_match_path:
                 print("Action requires a matching spec file, which was not found.")
                 input("Press Enter...")
            elif choice == 's' and not spec_match_path:
                 print("Cannot show spec details: No matching spec file found.")
                 input("Press Enter...")

            else:
                print(f"Invalid choice: {choice}")
                input("Press Enter...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Agent Network Analysis Browser")
    parser.add_argument("--eval", "-e", action="store_true", help="Evaluate all complete runs")
    parser.add_argument("--model", "-m", default=None, help="Model name to evaluate")
    parser.add_argument("--log-dir", "-d", default="logs", help="Path to the logs directory (default: logs)")
    parser.add_argument("--spec-dir", "-s", default=None, # Default to None, let the function find it
                        help="Path to the network spec files directory (optional; attempts auto-detection)")

    args = parser.parse_args()

    if args.eval:
        eval_all_complete_runs(
            args.log_dir,
            find_spec_dir(args.spec_dir),
            model_to_evaluate=args.model
        )
        sys.exit(0)

    # Ensure log dir exists or provide a helpful message
    if not os.path.isdir(args.log_dir):
        print(f"Error: Log directory '{os.path.abspath(args.log_dir)}' not found.")
        # Optionally offer to create it?
        # create = input(f"Create directory '{args.log_dir}'? (y/n): ")
        # if create.lower() == 'y':
        #     try:
        #         os.makedirs(args.log_dir)
        #         print(f"Created log directory: {args.log_dir}")
        #     except OSError as e:
        #         print(f"Error: Could not create log directory '{args.log_dir}'. {e}")
        #         sys.exit(1)
        # else:
        print("Please ensure the log directory exists or specify a correct path using --log-dir.")
        sys.exit(1)

    print(f"Starting browser in log directory: {os.path.abspath(args.log_dir)}")
    if args.spec_dir:
         print(f"Using specified spec directory: {os.path.abspath(args.spec_dir)}")
    print("Default view: Showing only the most recent run per model-spec pair.") # Inform user of default

    # Pass args directly to the browser function
    interactive_run_browser(
        args.log_dir, 
        args.spec_dir, 
        only_complete=True
    )


