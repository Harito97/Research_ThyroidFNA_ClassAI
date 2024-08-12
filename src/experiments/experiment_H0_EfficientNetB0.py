import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from src.models.efficient_net import H0_EfficientNetB0
from src.data.dataset.image_classification import MultiImageFolderDataset
from src.utils.build_model.image_classification import (
    TrainImageClassificationModel,
    ValidImageClassificationModel,
)
import yaml
import os


def run(config):
    print(f"Running Experiment {config['experiment']}: {config['name']}")

    # Set random seeds for reproducibility
    torch.manual_seed(config["seed"])

    # Data augmentation and resizing
    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize images to 224x224
            (
                transforms.RandomHorizontalFlip()
                if config["augmentation"]["horizontal_flip"]
                else transforms.Lambda(lambda x: x)
            ),
            (
                transforms.RandomVerticalFlip()
                if config["augmentation"]["vertical_flip"]
                else transforms.Lambda(lambda x: x)
            ),
            (
                transforms.RandomRotation(config["augmentation"]["rotation_range"])
                if config["augmentation"]["rotation_range"] > 0
                else transforms.Lambda(lambda x: x)
            ),
            transforms.ToTensor(),
            # Normalize if needed
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize images to 224x224
            transforms.ToTensor(),
            # Normalize if needed
        ]
    )

    # Load datasets
    train_dataset = MultiImageFolderDataset(
        experiment_yaml_config=config,
        root_dirs=config["data"]["train_path"], transform=train_transforms
    )
    val_dataset = MultiImageFolderDataset(
        experiment_yaml_config=config,
        root_dirs=config["data"]["val_path"], transform=val_transforms
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
    )
    print("Number of training samples:", len(train_dataset))
    print("Number of validation samples:", len(val_dataset))
    print("Data loaded successfully.")

    # Initialize model
    model = H0_EfficientNetB0(
        retrain_whole_net=config["model"]["retrain_whole_net"],
        dropout_rate=config["model"]["dropout_rate"], num_classes=len(config["classes"])
    )
    model.to(config["device"])

    # Calculate class weights for CrossEntropyLoss
    class_counts = torch.tensor(
        [train_dataset.get_labels().count(c) for c, _ in enumerate(config["classes"])]
    )
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    print("Class weights:", class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(config["device"]))

    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # StepLR: Decays the learning rate of each parameter group by gamma every step_size epochs.
    # When num_epochs = 100, step_size should be 10, gamma should be 0.1
    # When num_epochs = 50, step_size should be 5, gamma should be 0.1

    # Initialize training class
    trainer = TrainImageClassificationModel(
        experiment_yaml_config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config["training"]["epochs"],
        patience=config["training"]["patience"],
        device=config["device"],
        save_path=config["training"]["save_path"],
    )

    # Start training
    trainer.train()

    # Validate model
    validator = ValidImageClassificationModel(
        experiment_yaml_config=config,
        model=model,
        val_loader=val_loader,
    )

    # Start validation
    validator.evaluate()

    print(f"Experiment {config['experiment']} completed.")


if __name__ == "__main__":
    # Example of how to run the experiment
    # Load configuration from YAML file
    with open("configs/experiment_8.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Run the experiment
    run(config)
