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

async def run_job(model: str, network_spec: str, verbose_run: bool, logging_run: bool) -> Tuple[str, str, bool]:
    """Runs a single instance of run.py as a subprocess."""
    # Since launch.py is now in the src directory, run.py is in the same directory
    run_script_path = "run.py"
    cmd = [
        sys.executable,  # Use the current Python interpreter
        run_script_path,
        network_spec,
        "--model", model,
    ]
    if verbose_run:
        cmd.append("--verbose")
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

async def manager(jobs: List[Tuple[str, str]], global_limit: int, per_model_limits: Dict[str, int], verbose_run: bool, logging_run: bool):
    """Manages the concurrent execution of jobs respecting limits."""
    pending_jobs = collections.deque(jobs)
    running_tasks: Dict[asyncio.Task, str] = {} # Map task to model name
    running_per_model = collections.defaultdict(int)
    results = []

    while pending_jobs or running_tasks:
        # --- Check for completed tasks ---
        finished_tasks = {task for task in running_tasks if task.done()}
        for task in finished_tasks:
            model = running_tasks.pop(task) # Remove and get model
            running_per_model[model] -= 1
            try:
                _, network, success = await task # Get result
                results.append((model, network, success))
                print(f"[MANAGER] Completed: {model}/{network}. Remaining: {len(pending_jobs)} pending, {len(running_tasks)} running.")
            except Exception as e:
                print(f"[MANAGER] ERROR in task for model {model}: {e}")
                # Decide how to handle errors - maybe add to results as failure
                results.append((model, "unknown_network_due_to_error", False))


        # --- Try to launch new tasks ---
        while pending_jobs and len(running_tasks) < global_limit:
            model_to_run, network_to_run = pending_jobs[0] # Peek
            
            # Check per-model limit (use global if specific model not in per_model_limits)
            model_limit = per_model_limits.get(model_to_run, global_limit)
            
            if running_per_model[model_to_run] < model_limit:
                # Limits allow, launch the job
                model, network = pending_jobs.popleft() # Actually remove job
                
                task = asyncio.create_task(run_job(model, network, verbose_run, logging_run))
                running_tasks[task] = model # Track task and its model
                running_per_model[model] += 1
                print(f"[MANAGER] Launched: {model}/{network}. Running: {len(running_tasks)} total, {running_per_model[model]} for {model}. Pending: {len(pending_jobs)}.")
            else:
                # Cannot launch this job due to per-model limit, try next (if any) or wait
                # If we checked the first pending job and couldn't run it, we probably can't run others of the same model either.
                # Need to wait for a slot for this model to free up.
                break 

        # --- Wait if needed ---
        if pending_jobs and len(running_tasks) >= global_limit:
            # Global limit reached, must wait for *any* task to finish
            if running_tasks:
                _, pending = await asyncio.wait(running_tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
        elif pending_jobs and any(running_per_model[pending_jobs[0][0]] >= per_model_limits.get(pending_jobs[0][0], global_limit) for i in range(len(pending_jobs)) if i==0):
             # Cannot launch the next job due to per-model limit, wait for a relevant task to finish
             model_to_wait_for = pending_jobs[0][0]
             tasks_for_model = {task for task, model in running_tasks.items() if model == model_to_wait_for}
             if tasks_for_model:
                 _, pending = await asyncio.wait(tasks_for_model, return_when=asyncio.FIRST_COMPLETED)
             elif not running_tasks: # Should not happen if pending_jobs is not empty, but safety check
                 await asyncio.sleep(0.1) # Avoid busy loop if something is weird
             else: # No running tasks for the model needed, but global limit not hit -> wait for any task
                 _, pending = await asyncio.wait(running_tasks.keys(), return_when=asyncio.FIRST_COMPLETED)

        else:
            # No jobs pending, just wait for remaining running tasks
            if running_tasks:
                 _, pending = await asyncio.wait(running_tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
            else:
                 # All done
                 break

        # Small sleep to prevent potential busy-looping in edge cases and allow event loop to cycle
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
    parser.add_argument("--config", "-c", type=str, help="Path to YAML configuration file.")
    
    # Direct CLI args (override config file)
    parser.add_argument("--models", type=str, help="Comma-separated list of models (e.g., 'model1,model2').")
    parser.add_argument("--networks", type=str, help="Comma-separated list of network spec names (e.g., 'spec1,spec2').")
    parser.add_argument("--global-max", type=int, help="Maximum total concurrent jobs.")
    parser.add_argument("--model-max", type=str, 
                        help="Per-model max concurrent jobs (e.g., 'gemini=3,claude=2'). Unspecified models use global max.")

    # Passthrough args for run.py
    parser.add_argument("--verbose-run", action="store_true", default=False, help="Pass --verbose to each run.py instance.")
    parser.add_argument("--logging-run", action="store_true", default=False, help="Pass --logging to each run.py instance.")

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
    networks_list = [n.strip() for n in networks_list if n.strip()]

    # Generate all job combinations
    jobs_to_run = [(model, network) for model in models_list for network in networks_list]

    print("[LAUNCHER] Configuration:")
    print(f"  Models: {models_list}")
    print(f"  Networks: {networks_list}")
    print(f"  Total Jobs: {len(jobs_to_run)}")
    print(f"  Global Max Concurrent: {global_max}")
    print(f"  Per-Model Max Concurrent: {per_model_max if per_model_max else 'Same as global'}")
    print(f"  Pass --verbose to run.py: {args.verbose_run}")
    print(f"  Pass --logging to run.py: {args.logging_run}")


    # Run the manager
    try:
        asyncio.run(manager(jobs_to_run, global_max, per_model_max, args.verbose_run, args.logging_run))
    except KeyboardInterrupt:
        print("\n[LAUNCHER] Keyboard interrupt received. Shutting down...")
        # Note: Currently running subprocesses might continue until finished.
        # More sophisticated handling could involve sending SIGTERM to children.

if __name__ == "__main__":
    # No need to check for file existence since we're in the same directory
    main() 