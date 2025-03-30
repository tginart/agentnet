import argparse
import asyncio
import collections
import yaml
import os
import sys
from typing import List, Dict, Tuple, Any

# --- Configuration Defaults ---
DEFAULT_GLOBAL_MAX_CONCURRENT = 3
DEFAULT_PER_MODEL_MAX_CONCURRENT = {} # Empty dict means use global max for all models

# --- Helper Functions ---

def parse_model_max(value: str) -> Dict[str, int]:
    """Parses the per-model max string (e.g., 'gemini=3,claude=2')."""
    limits = {}
    if not value:
        return limits
    try:
        pairs = value.split(',')
        for pair in pairs:
            model, limit_str = pair.strip().split('=')
            limits[model.strip()] = int(limit_str.strip())
        return limits
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid format for model-max: '{value}'. Use 'model1=N,model2=M'. Error: {e}")

async def run_job(model: str, network_spec: str, logging_run: bool) -> Tuple[str, str, bool]:
    """Runs a single instance of run.py as a subprocess."""
    run_script_path = "run.py"  # Now directly in the root directory
    cmd = [
        sys.executable,  # Use the current Python interpreter
        run_script_path,
        network_spec,
        "--model", model,
    ]
    if logging_run:
        cmd.append("--logging")

    print(f"[LAUNCHER] Starting job: Model='{model}', Network='{network_spec}' (CMD: {' '.join(cmd)})")
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await process.communicate()
    success = process.returncode == 0
    
    print(f"[LAUNCHER] Finished job: Model='{model}', Network='{network_spec}', Success={success}, ReturnCode={process.returncode}")
    if not success:
        print(f"[LAUNCHER]   Stderr for {model}/{network_spec}:\n{stderr.decode().strip()}")
    # Uncomment below if you want to see stdout even for successful runs
    # else: 
    #    print(f"[LAUNCHER]   Stdout for {model}/{network_spec}:\n{stdout.decode().strip()}")

    return model, network_spec, success

# --- Relaunch Helper ---

def check_if_completed(model: str, network: str, log_base_dir: str = "logs") -> bool:
    """Checks if a completed log directory exists for the given model and network.

    Scans the log_base_dir for subdirectories and examines metadata files within each:
    - config.yaml for the model name
    - spec_path.txt for the network spec name
    - network.json to verify completion
    """
    if not os.path.isdir(log_base_dir):
        print(f"[LAUNCHER] Log directory '{log_base_dir}' not found.")
        return False # Log directory doesn't exist, so no jobs are completed
    
    try:
        for entry in os.listdir(log_base_dir):
            full_path = os.path.join(log_base_dir, entry)
            
            if not os.path.isdir(full_path):
                continue  # Skip non-directories
                
            # First check if this looks like a completed run
            network_json = os.path.join(full_path, "network.json")
            if not os.path.isfile(network_json):
                continue  # Skip directories without network.json (incomplete runs)
            
            # Check config.yaml for model name
            config_file = os.path.join(full_path, "config.yaml")
            if not os.path.isfile(config_file):
                continue  # Skip if no config.yaml
                
            # Check for spec_path.txt for network name
            spec_path_file = os.path.join(full_path, "spec_path.txt")
            if not os.path.isfile(spec_path_file):
                continue  # Skip if no spec_path.txt
            
            # Now read both files to check for matches
            try:
                # Check model name from config.yaml
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                    if not (config_data and isinstance(config_data, dict) and 'model' in config_data):
                        continue  # Skip if config.yaml doesn't have model
                    
                    config_model = config_data['model']
                    if config_model != model:
                        continue  # Skip if model doesn't match
                
                # Check network name from spec_path.txt
                with open(spec_path_file, 'r') as f:
                    spec_path = f.read().strip()
                    # Extract the network name from the spec path
                    # Assuming spec_path.txt contains a path ending with the network name
                    spec_network = os.path.basename(spec_path).replace('.json', '')
                    if spec_network != network:
                        continue  # Skip if network doesn't match
                
                # If we got here, both model and network match!
                print(f"[LAUNCHER] Found completed run for {model}/{network} in {full_path}")
                return True
                
            except yaml.YAMLError as e:
                print(f"[LAUNCHER] Warning: Error parsing YAML file '{config_file}': {e}")
                continue
            except Exception as e:
                print(f"[LAUNCHER] Warning: Error processing files in '{full_path}': {e}")
                continue

    except OSError as e:
        print(f"[LAUNCHER] Warning: Error scanning log directory '{log_base_dir}': {e}")
        return False

    return False # No matching completed job found

async def manager(jobs: List[Tuple[str, str]], global_limit: int, per_model_limits: Dict[str, int], logging_run: bool):
    """Manages the concurrent execution of jobs respecting limits."""
    pending_jobs = collections.deque(jobs)
    running_tasks: Dict[asyncio.Task, str] = {} # Map task to model name
    running_per_model = collections.defaultdict(int)
    results = []

    while pending_jobs or running_tasks:
        # --- Check for completed tasks ---
        finished_tasks = {task for task in running_tasks if task.done()}
        any_task_finished = bool(finished_tasks) # Track if any task finished this cycle
        for task in finished_tasks:
            model = running_tasks.pop(task) # Remove and get model
            running_per_model[model] -= 1
            try:
                # Ensure await task happens before accessing results to handle exceptions
                task_result = await task
                _, network, success = task_result # Get result
                results.append((model, network, success))
                print(f"[MANAGER] Completed: {model}/{network}. Success: {success}. Remaining: {len(pending_jobs)} pending, {len(running_tasks)} running.")
            except Exception as e:
                # Attempt to find the network associated with the failed task if possible
                # This might require storing the (model, network) tuple with the task
                # For now, mark network as unknown
                network_name = "unknown_network_due_to_error" 
                results.append((model, network_name, False))
                print(f"[MANAGER] ERROR in completed task for model {model} (Network: {network_name}): {e}")


        # --- Try to launch new tasks ---
        launched_a_job_in_this_cycle = False
        keep_trying_to_launch = True # Flag to control the inner launch loop
        while keep_trying_to_launch and pending_jobs and len(running_tasks) < global_limit:
            keep_trying_to_launch = False # Assume we won't find a job to launch in this pass
            
            found_job_to_launch_at_index = -1
            # Find the *first* index in the current deque that can be launched
            for i in range(len(pending_jobs)):
                model_to_run, _ = pending_jobs[i] # Only need model to check limit
                model_limit = per_model_limits.get(model_to_run, global_limit)
                
                # Check if the model is under its specific limit
                if running_per_model[model_to_run] < model_limit:
                    # Found a runnable job!
                    found_job_to_launch_at_index = i
                    break # Stop searching, we'll launch this one

            if found_job_to_launch_at_index != -1:
                # A job can be launched. Rotate deque to bring it to the front.
                pending_jobs.rotate(-found_job_to_launch_at_index)
                model, network = pending_jobs.popleft() # Pop the runnable job
                
                # Launch the job
                task = asyncio.create_task(run_job(model, network, logging_run))
                running_tasks[task] = model # Track task and its model
                running_per_model[model] += 1
                print(f"[MANAGER] Launched: {model}/{network}. Running: {len(running_tasks)} total, {running_per_model[model]} for {model}. Pending: {len(pending_jobs)}.")
                
                launched_a_job_in_this_cycle = True # Mark that we launched something overall
                keep_trying_to_launch = True # Since we launched one, try immediately for another
            
            # If found_job_to_launch_at_index remains -1, no runnable job was found in the deque
            # keep_trying_to_launch stays False, and the inner while loop terminates.

        # --- Wait if needed ---
        needs_to_wait = False
        # Condition 1: Jobs pending, but we made no progress (didn't launch, and nothing finished)
        if pending_jobs and not launched_a_job_in_this_cycle and not any_task_finished:
             needs_to_wait = True
        # Condition 2: No jobs pending, but tasks are still running
        elif not pending_jobs and running_tasks:
             needs_to_wait = True
        # Condition 3: Safety check - pending jobs but no running tasks (should ideally not happen)
        elif pending_jobs and not running_tasks:
             needs_to_wait = True
             print("[MANAGER] Warning: Pending jobs but no running tasks. Waiting.")


        if needs_to_wait and running_tasks:
             wait_tasks = running_tasks.keys() # Default: wait for any task if unsure

             # If global limit isn't hit, try to wait more specifically
             if len(running_tasks) < global_limit:
                  # Blocked by per-model limits. Find relevant tasks to wait for.
                  blocking_models = set()
                  tasks_for_blocking_models = set()

                  # Identify models that are at their limit AND needed by pending jobs
                  for i in range(len(pending_jobs)):
                       model_to_run, _ = pending_jobs[i]
                       model_limit = per_model_limits.get(model_to_run, global_limit)
                       if running_per_model[model_to_run] >= model_limit:
                           blocking_models.add(model_to_run)
                  
                  # Collect running tasks associated with these blocking models
                  if blocking_models:
                      tasks_for_blocking_models = {
                          task for task, model in running_tasks.items() if model in blocking_models
                      }
                  
                  # Only wait for specific tasks if we found relevant ones
                  if tasks_for_blocking_models:
                       wait_tasks = tasks_for_blocking_models
                       print(f"[MANAGER] Blocked by per-model limits ({blocking_models}). Waiting for one of {len(wait_tasks)} relevant tasks.")
                  else:
                       # This case might occur if a model limit is blocking, but the last task for that model *just* finished
                       # Or if limits/counts are somehow inconsistent. Waiting for any task is safer fallback.
                       print(f"[MANAGER] Blocked, but no specific running tasks found for blocking models ({blocking_models}). Waiting for any task.")

             else:
                  # Global limit reached or exceeded, wait for any task.
                  print(f"[MANAGER] Global limit ({global_limit}) reached. Waiting for any task.")

             # Perform the wait
             _, pending = await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)

        elif not running_tasks and not pending_jobs:
             # All done! Exit the main loop.
             break
        
        # Small sleep to prevent potential high CPU usage in edge cases or if waiting without tasks
        if not any_task_finished and not launched_a_job_in_this_cycle:
             await asyncio.sleep(0.05) 


    print("\n[LAUNCHER] All jobs completed.")
    # Summarize results
    success_count = sum(1 for _, _, success in results if success)
    fail_count = len(results) - success_count
    print(f"[LAUNCHER] Summary: {success_count} succeeded, {fail_count} failed.")
    if fail_count > 0:
        print("[LAUNCHER] Failed jobs:")
        for model, network, success in results:
            if not success:
                print(f"  - Model: {model}, Network: {network}")


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Launch multiple run.py processes concurrently.")
    
    # Config File
    parser.add_argument("--config", "-c", type=str, default="configs/launch_config.yaml", help="Path to YAML configuration file.")
    
    # Direct CLI args (override config file)
    parser.add_argument("--models", type=str, help="Comma-separated list of models (e.g., 'model1,model2').")
    parser.add_argument("--networks", type=str, help="Comma-separated list of network spec names (e.g., 'spec1,spec2').")
    parser.add_argument("--global-max", type=int, help="Maximum total concurrent jobs.")
    parser.add_argument("--model-max", type=str, 
                        help="Per-model max concurrent jobs (e.g., 'gemini=3,claude=2'). Unspecified models use global max.")
    
    # Relaunch flag
    parser.add_argument("--relaunch", action="store_true", default=False,
                        help="Skip jobs that have already completed (indicated by logs/<model>/<network>/network.json existing).")

    # Passthrough args for run.py
    parser.add_argument("--logging-run", "--log", "-l", action="store_true", default=False, help="Pass --logging to each run.py instance.")

    args = parser.parse_args()

    # Load config from YAML if specified
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            print(f"[LAUNCHER] Loaded configuration from {args.config}")
        except FileNotFoundError:
            print(f"[LAUNCHER] Error: Config file '{args.config}' not found.")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"[LAUNCHER] Error parsing YAML file '{args.config}': {e}")
            sys.exit(1)

    # Determine final configuration (CLI overrides YAML)
    models_list = args.models.split(',') if args.models else config.get('models', [])
    networks_list = args.networks.split(',') if args.networks else config.get('networks', [])
    global_max = args.global_max if args.global_max is not None else config.get('global_max', DEFAULT_GLOBAL_MAX_CONCURRENT)
    
    # Per-model limits: start with default, update from YAML, then update from CLI
    per_model_max = DEFAULT_PER_MODEL_MAX_CONCURRENT.copy()
    per_model_max.update(config.get('per_model_max', {}))
    if args.model_max:
        try:
             per_model_max.update(parse_model_max(args.model_max))
        except argparse.ArgumentTypeError as e:
             print(f"[LAUNCHER] Error parsing --model-max argument: {e}")
             sys.exit(1)

    # Validate configuration
    if not models_list:
        print("[LAUNCHER] Error: No models specified either via --models or config file.")
        sys.exit(1)
    if not networks_list:
        print("[LAUNCHER] Error: No networks specified either via --networks or config file.")
        sys.exit(1)
    if global_max <= 0:
        print("[LAUNCHER] Error: --global-max must be positive.")
        sys.exit(1)
    
    # Clean up lists
    models_list = [m.strip() for m in models_list if m.strip()]
    #breakpoint()
    networks_list = [n.strip() for n in networks_list if n.strip()]

    # Generate all potential job combinations
    all_potential_jobs = [(model, network) for model in models_list for network in networks_list]
    jobs_to_run = []
    skipped_jobs = []

    # Filter jobs if --relaunch is enabled
    if args.relaunch:
        print("[LAUNCHER] --relaunch enabled: Checking for completed jobs in 'logs/' directory...")
        for model, network in all_potential_jobs:
            if check_if_completed(model, network):
                skipped_jobs.append((model, network))
            else:
                jobs_to_run.append((model, network))

        if skipped_jobs:
            print(f"[LAUNCHER] Skipped {len(skipped_jobs)} completed jobs (found logs/.../network.json):")
            # Limit printing if too many? For now, print all skipped.
            for model, network in skipped_jobs:
                 print(f"  - Skipped: {model}/{network}")
            print("-" * 20) # Separator
        else:
             print("[LAUNCHER] No previously completed jobs found to skip.")
    else:
        jobs_to_run = all_potential_jobs


    print("[LAUNCHER] Configuration:")
    print(f"  Models: {models_list}")
    print(f"  Networks: {networks_list}")
    print(f"  Total Potential Jobs: {len(all_potential_jobs)}") # Show total before skipping
    if args.relaunch:
        print(f"  Jobs Skipped (Completed): {len(skipped_jobs)}")
    print(f"  Jobs To Run Now: {len(jobs_to_run)}") # Show actual count
    print(f"  Global Max Concurrent: {global_max}")
    print(f"  Per-Model Max Concurrent: {per_model_max if per_model_max else 'Same as global'}")
    print(f"  Pass --logging to run.py: {args.logging_run}")

    # Exit early if no jobs to run
    if not jobs_to_run:
        print("[LAUNCHER] No jobs remaining to run. Exiting.")
        sys.exit(0)

    # Run the manager
    try:
        asyncio.run(manager(jobs_to_run, global_max, per_model_max, args.logging_run))
    except KeyboardInterrupt:
        print("\n[LAUNCHER] Keyboard interrupt received. Shutting down...")
        # Note: Currently running subprocesses might continue until finished.
        # More sophisticated handling could involve sending SIGTERM to children.

if __name__ == "__main__":
    main() 