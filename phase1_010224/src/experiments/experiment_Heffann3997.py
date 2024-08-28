import torch
import yaml
import os
from src.models.heffann3997 import Heffann3997
from tests.test_result_of_heffann3997.tool_valid import Validator

def run(config):
    print(f"Running Experiment {config['experiment']}: {config['name']}")

    # Set random seeds for reproducibility
    torch.manual_seed(config["seed"])

    # Initialize model
    model = Heffann3997()
    model.load_weight(
        module_1_path=config["model"]["module1"],
        module_2_path=config["model"]["module2"],
    )
    model.to(config["device"])

    # Validate model
    validator = Validator(
        experiment_yaml_config=config,
        model=model,
        data_dir=config["data"]["train_path"][0],
    )

    # Start validation
    print("Starting validation on train set of dataver0...")
    validator.evaluate()

    # Validate model
    validator = Validator(
        experiment_yaml_config=config,
        model=model,
        data_dir=config["data"]["val_path"][0],
    )

    # Start validation
    print("Starting validation on valid set of dataver0...")
    validator.evaluate()

    print(f"Experiment {config['experiment']} completed.")


if __name__ == "__main__":
    # Example of how to run the experiment
    # Load configuration from YAML file
    with open("configs/experiment_??.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Run the experiment
    run(config)
