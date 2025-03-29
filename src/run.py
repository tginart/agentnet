import argparse
import asyncio
import os
import json
import yaml

from sim.network_runner import NetworkRunner, RunConfig
from sim.network_analysis import NetworkLogger

async def main(spec_name, run_config, enable_logging=False):
    # convert run_config to RunConfig object
    run_config = RunConfig(**run_config)

    # Construct the full path to the spec file
    spec_path = os.path.join(os.path.dirname(__file__), "sim/network_specs", spec_name)

    # spec_path does not end in .json, add it
    if not spec_path.endswith(".json"):
        spec_path += ".json"
    
    # Check if the file exists
    if not os.path.exists(spec_path):
        print(f"Error: Spec file '{spec_path}' not found.")
        return
    
    # load spec from file
    with open(spec_path, "r") as f:
        spec = json.load(f)
    
    # Set up logger if enabled
    logger = None
    if enable_logging:
        logger = NetworkLogger()
        print(f"Logging enabled, logs will be saved to {logger.log_file}")
    
    # Create the network runner with logger if logging is enabled
    print(f"Loading network from {spec_path}...")
    runner = NetworkRunner(spec_path, run_config=run_config, logger=logger)

    print(f"Running network with task: {spec['task']}")
    result = await runner.run_network(spec['task'])
    
    # Save network structure if logging is enabled
    if enable_logging:
        network_file = logger.save_network_structure()
        print(f"Network communication graph saved to {network_file}")
    
    print("\nFinal result:")
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an agent network from a JSON spec file")
    parser.add_argument("spec_name", help="Name of the JSON spec file in the network_specs directory")
    parser.add_argument("--config", "-c", type=str, default="configs/run_config.yaml",
                        help="Path to the run configuration file")
    parser.add_argument("--print", "-p", action="store_true", default=False,
                        help="Print logs to stdout")
    parser.add_argument("--model", "-m", default="claude-3-5-sonnet-20240620",
                        help="Model to use for the simulation")
    parser.add_argument("--logging", "-l", action="store_true", default=False,
                        help="Enable logging of network communication")
    

    args = parser.parse_args()

    # override config file with cli args if provided
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    if args.model:
        config["model"] = args.model
    if args.print:
        config["stdout"] = args.print

    breakpoint()
    
    asyncio.run(main(args.spec_name, config, args.logging))
