import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.models.heffann3997 import Heffann3997
from src.data.dataset.image_classification import MultiImageFolderDataset
from src.utils.build_model.image_classification import (
    ValidImageClassificationModel,
)
import yaml
import os


def run(config):
    print(f"Running Experiment {config['experiment']}: {config['name']}")

    # Set random seeds for reproducibility
    torch.manual_seed(config["seed"])

    # Transform
    val_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize images to 224x224
            transforms.ToTensor(),
        ]
    )

    # Load datasets
    train_dataset = MultiImageFolderDataset(
        experiment_yaml_config=config,
        root_dirs=config["data"]["train_path"],
        transform=val_transforms,
    )
    val_dataset = MultiImageFolderDataset(
        experiment_yaml_config=config,
        root_dirs=config["data"]["val_path"],
        transform=val_transforms,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
    )
    print("Number of training samples:", len(train_dataset))
    print("Number of validation samples:", len(val_dataset))
    print("Data loaded successfully.")

    # Initialize model
    model = Heffann3997()
    model.load_weight(
        module_1_path=config["model"]["module1"],
        module_2_path=config["model"]["module2"],
    )
    model.to(config["device"])

    # Validate model
    validator = ValidImageClassificationModel(
        experiment_yaml_config=config,
        model=model,
        val_loader=train_loader,
    )

    # Start validation
    print("Starting validation on train set of dataver0...")
    validator.evaluate()

    # Validate model
    validator = ValidImageClassificationModel(
        experiment_yaml_config=config,
        model=model,
        val_loader=val_loader,
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
