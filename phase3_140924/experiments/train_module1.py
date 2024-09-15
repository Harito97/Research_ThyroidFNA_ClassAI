import yaml
import argparse
import os
from src.models.module1.cnn import get_cnn_model
from src.models.module1.vit import get_vit_model
from src.data.data_loader import get_dataloader
from src.training.train_model import TrainClassificationModel
from src.utils.helpers import load_config

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
import numpy as np
import random


def initialize_model(model_type, num_classes):
    """
    Khởi tạo mô hình phân loại theo loại mô hình yêu cầu.

    Args:
        model_type (str): Loại mô hình (CNN, ViT).
        num_classes (int): Số lượng lớp (output classes).

    Returns:
        nn.Module: Mô hình đã được khởi tạo.
    """
    model_map = {
        "vgg16": "vgg16", "vgg19": "vgg19",
        "resnet18": "resnet18", "resnet152": "resnet152",
        "densenet121": "densenet121", "densenet201": "densenet201",
        "efficientnet_b0": "efficientnet_b0", "efficientnet_b7": "efficientnet_b7",
        "mobilenet_v1": "mobilenet_v1", "mobilenet_v3_large": "mobilenet_v3_large",
        "vit_b_16": "vit_b_16", "vit_l_16": "vit_l_16"
    }
    
    if model_type in model_map:
        if "vit" in model_type:
            return get_vit_model(name=model_map[model_type], num_classes=num_classes)
        else:
            return get_cnn_model(name=model_map[model_type], num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def setup_data_loaders(train_dir, val_dir, batch_size, augmentations):
    """
    Thiết lập DataLoader cho tập train và validation.

    Args:
        train_dir (str): Đường dẫn đến thư mục train.
        val_dir (str): Đường dẫn đến thư mục validation.
        batch_size (int): Kích thước batch.
        augmentations (dict): Cấu hình augmentation.

    Returns:
        tuple: DataLoader cho tập train và validation.
    """
    # Augmentation cho tập train
    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip() if augmentations["horizontal_flip"] else transforms.Lambda(lambda x: x),
            transforms.RandomVerticalFlip() if augmentations["vertical_flip"] else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(augmentations["rotation"]) if augmentations["rotation"] > 0 else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
        ]
    )
    
    # Augmentation cho tập validation
    val_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # Tạo DataLoader cho tập train và validation
    train_loader = get_dataloader(
        root_dir=train_dir,
        batch_size=batch_size,
        shuffle=True,
        transform=train_transforms
    )
    
    val_loader = get_dataloader(
        root_dir=val_dir,
        batch_size=batch_size,
        shuffle=False,
        transform=val_transforms
    )

    return train_loader, val_loader


def calculate_class_weights(train_loader, num_classes):
    """
    Tính toán trọng số cho từng class dựa trên tần suất xuất hiện trong tập train.

    Args:
        train_loader (DataLoader): DataLoader cho tập train.
        num_classes (int): Số lượng lớp.

    Returns:
        Tensor: Trọng số của từng class.
    """
    class_counts = torch.zeros(num_classes)
    
    # Duyệt qua toàn bộ các batch trong train_loader để đếm số lượng từng class
    for _, labels in train_loader:
        for label in labels:
            class_counts[label] += 1
    
    # Tính trọng số ngược lại với số lượng class
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    
    print("Class weights:", class_weights)
    return class_weights


def setup_training_components(model, train_loader, num_classes, device, lr):
    """
    Thiết lập tiêu chuẩn mất mát, optimizer và scheduler cho huấn luyện.

    Args:
        model (nn.Module): Mô hình đã được khởi tạo.
        train_loader (DataLoader): DataLoader của tập train.
        num_classes (int): Số lượng lớp.
        device (str): Thiết bị (CPU hoặc GPU).
        lr (float): Learning rate.

    Returns:
        tuple: Criterion, optimizer, scheduler.
    """
    # Tính toán class weights và thiết lập criterion
    class_weights = calculate_class_weights(train_loader, num_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Thiết lập optimizer và scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    return criterion, optimizer, scheduler


def train_module1(config_path):
    """
    Hàm chính để huấn luyện mô hình phân loại ảnh dựa trên cấu hình từ file YAML.

    Args:
        config_path (str): Đường dẫn tới file cấu hình YAML.
    """
    # Load cấu hình
    config = load_config(config_path)

    # Đặt seed để đảm bảo tính nhất quán
    seed = config["trainer"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Thiết lập thông số từ cấu hình
    model_type = config["trainer"]["model_type"]
    num_epochs = config["trainer"].get("num_epochs", 10)
    batch_size = config["trainer"].get("batch_size", 32)
    patience = config["trainer"].get("patience", 10)
    device = config["trainer"].get("device", "cpu") if torch.cuda.is_available() else "cpu"
    lr = config["trainer"].get("lr", 0.0001)
    num_classes = config["data"]["num_classes"]
    train_dir = config["data"]["train_dir"]
    val_dir = config["data"]["val_dir"]
    augmentations = config["data"]["augmentation"]

    # Khởi tạo mô hình
    model = initialize_model(model_type, num_classes)

    # Thiết lập DataLoader cho train và validation
    train_loader, val_loader = setup_data_loaders(train_dir, val_dir, batch_size, augmentations)

    # Thiết lập tiêu chuẩn mất mát, optimizer và scheduler
    criterion, optimizer, scheduler = setup_training_components(model, train_loader, num_classes, device, lr)

    # Khởi tạo và cấu hình huấn luyện
    trainer = TrainClassificationModel(
        config_path=None,  # Không dùng lại file config ở đây
        model_type=model_type,
        num_epochs=num_epochs,
        patience=patience,
        device=device,
    )

    # Setup và huấn luyện mô hình
    trainer.setup(model, train_loader, val_loader, criterion, optimizer, scheduler)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình phân loại ảnh")
    parser.add_argument("--config_path", type=str, required=True, help="Đường dẫn tới file cấu hình YAML")
    args = parser.parse_args()

    # Gọi hàm train
    train_module1(config_path=args.config_path)
