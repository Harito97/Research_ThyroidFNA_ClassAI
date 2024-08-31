import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from src.models.module1.efficient_net import H0_EfficientNetB0
from src.data.dataset.image_dataset import MultiImageFolderDataset
from src.utils.evaluate import EvaluateClassificationModel
import yaml
import os
import time


def run(config):
    start_time = time.time()
    print(f"Running Experiment {config['experiment']}: {config['name']}")

    # Set random seeds for reproducibility
    torch.manual_seed(config["evaluate"]["seed"])

    transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize images to 224x224
            transforms.ToTensor(),
        ]
    )

    # Load datasets
    evaluate_dataset = MultiImageFolderDataset(
        config=config,
        root_dirs=config["data"]["valid_path"],
        transform=transforms,
    )

    # Create data loaders
    eval_loader = DataLoader(
        evaluate_dataset,
        batch_size=config["evaluate"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
    )

    print("Number of evaluate samples:", len(evaluate_dataset))
    print("Data loaded successfully.")

    # Initialize model
    model = H0_EfficientNetB0(
        num_classes=len(config["data"]["classes"]),
    )
    model.to(config["evaluate"]["device"])
    dir_path = config["info_read"]["dir_path"].format(
        time_stamp=config["evaluate"]["time_stamp"],
        model_name=config["evaluate"]["model_name"],
        model_type=config["evaluate"]["model_type"],
    )
    model_path = config["evaluate"]["model_path"].format(
        dir_path=dir_path, model_path=config["info_read"]["model_path"]
    )
    model.load_state_dict(
        torch.load(model_path, map_location=config["evaluate"]["device"])
    )
    model.eval()

    print("Evaluate model...")
    # Validate model
    evaluate = EvaluateClassificationModel(
        config=config,
        model=model,
        eval_loader=eval_loader,
    )
    # Start validation
    evaluate.evaluate()
    print("Evaluate completed.")

    print(f"Experiment {config['experiment']} completed in {time.time() - start_time}s.")


if __name__ == "__main__":
    # Example of how to run the experiment
    # Load configuration from YAML file
    with open("configs/experiment_??.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Run the experiment
    run(config)
