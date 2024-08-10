# Standard imports
import os
import argparse
from typing import Dict, Any

# External imports
import yaml

# Internal imports
from src.experiments import *  # Import all experiments
from src.data import data_creator, data_explore  # Import specific modules


def valid_file(param: str) -> str:
    if not os.path.isfile(param):
        raise argparse.ArgumentTypeError(f"File '{param}' does not exist")
    if not os.access(param, os.R_OK):
        raise argparse.ArgumentTypeError(f"File '{param}' is not readable")
    return param


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and return the configuration from a YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def run_task(config: Dict[str, Any]) -> None:
    """Run the appropriate task based on the configuration."""
    task_type = config.get("type")
    if task_type == "data_creator":
        data_creator.run(config=config)
    elif task_type == "data_explore":
        data_explore.run(config=config)
    elif task_type == "experiment":
        # Placeholder for experiment runner
        # experiment_runner.run(config)
        pass
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def main() -> None:
    """Main function to parse arguments and run the specified task."""
    parser = argparse.ArgumentParser(
        description="""Run: 
        1. Initialize datasets 
        2. Explore and analyze datasets 
        3. Medical image classification experiment """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=valid_file,
        required=True,
        metavar="PATH",
        help="Path to the configuration YAML file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_task(config)


if __name__ == "__main__":
    main()

# Example usage:
# python main.py --config configs/experiment_1.yaml
