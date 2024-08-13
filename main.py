# Standard imports
import os
import argparse
from typing import Dict, Any

# External imports
import yaml

# Internal imports
## Import all experiments
import src.experiments.experiment_H97_9_7_EfficientNetB0 as experiment_H97_9_7_EfficientNetB0    
import src.experiments.experiment_H97_9_7_EfficientNetB7 as experiment_H97_9_7_EfficientNetB7
import src.experiments.experiment_H0_EfficientNetB0 as experiment_H0_EfficientNetB0  
import src.experiments.experiment_H0_EfficientNetB7 as experiment_H0_EfficientNetB7
## Import all data modules
from src.data import data_creator, data_explore


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
        if config["experiment"] in ["1", "2", "3", "4", "5", "6"]: # Experiment 1 to 6
            experiment_H97_9_7_EfficientNetB0.run(config) # use model H97_9_7_EfficientNetB0, batch_size = 144
        elif config["experiment"] in ["8", "9", "10", "11", "12", "13"]: # Experiment 8 to 13
            experiment_H0_EfficientNetB0.run(config) # use model H0_EfficientNetB0, batch_size = 144
        elif config["experiment"] in ["15", "16", "17", "18", "19", "20"]: # Experiment 15 to 20
            experiment_H97_9_7_EfficientNetB0.run(config) # use model H97_9_7_EfficientNetB0, batch_size = 32
        elif config["experiment"] in ["22", "23", "24", "25", "26", "27"]: # Experiment 22 to 27
            experiment_H0_EfficientNetB0.run(config) # use model H0_EfficientNetB0, batch_size = 32
        elif config["experiment"] in ["29", "30", "31", "32", "33", "34"]: # Experiment 29 to 34
            experiment_H0_EfficientNetB7.run(config) # use model H0_EfficientNetB7, batch_size = 32
        elif config["experiment"] in ["36", "37", "38", "39", "40", "41"]: # Experiment 36 to 41
            experiment_H97_9_7_EfficientNetB7.run(config) # use model H97_9_7_EfficientNetB7, batch_size = 32 vs 16 if retrains whole network
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
