# experiments/test_module1.py
import argparse
import sys
import os
import torch
from torchvision import transforms

# Thêm thư mục gốc của dự án vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.module1.cnn import get_cnn_model
from src.models.module1.vit import get_vit_model
from src.data.data_loader import get_dataloader
from src.utils.utils import load_config
from src.training.test_model import TestClassificationModel

# from src.utils.helpers import load_criterion  # not defined yet


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
        "vgg16": "vgg16",
        "vgg19": "vgg19",
        "resnet18": "resnet18",
        "resnet152": "resnet152",
        "densenet121": "densenet121",
        "densenet201": "densenet201",
        "efficientnet_b0": "efficientnet_b0",
        "efficientnet_b7": "efficientnet_b7",
        "mobilenet_v1": "mobilenet_v1",
        "mobilenet_v3_large": "mobilenet_v3_large",
        "vit_b_16": "vit_b_16",
        "vit_l_16": "vit_l_16",
    }

    if model_type in model_map:
        if "vit" in model_type:
            return get_vit_model(name=model_map[model_type], num_classes=num_classes)
        else:
            return get_cnn_model(name=model_map[model_type], num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def load_model_weights(model, model_path):
    """
    Load trọng số cho mô hình.

    Args:
        model (nn.Module): Mô hình cần load trọng số.
        model_path (str): Đường dẫn đến file trọng số.

    Returns:
        nn.Module: Mô hình đã được load trọng số.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # map_location=torch.device('cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def setup_data_loaders(test_dir, batch_size):
    """
    Thiết lập DataLoader cho tập test.

    Args:
        test_dir (str): Đường dẫn đến thư mục test.
        batch_size (int): Kích thước batch.

    Returns:
        tuple: DataLoader cho tập test.
    """
    # Augmentation cho tập test
    test_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # Tạo DataLoader cho tập test
    test_loader = get_dataloader(
        root_dir=test_dir,
        batch_size=batch_size,
        shuffle=False,
        transform=test_transforms,
    )

    return test_loader


def test_module1(config_path):
    config = load_config(config_path)

    # Load mô hình đã được huấn luyện
    model = initialize_model(model_type=config["tester"]["model_type"], num_classes=3)
    model = load_model_weights(model=model, model_path=config["tester"]["model_path"])

    # Load dataloader của tập test
    test_loader = setup_data_loaders(
        test_dir=config["tester"]["test_data_path"],
        batch_size=config["tester"]["batch_size"],
    )

    # Load tiêu chí tính loss
    # criterion = load_criterion(config["tester"]["criterion"]) # no need use criterion when testing

    # Thiết lập và kiểm tra mô hình
    tester = TestClassificationModel(config=config)
    tester.setup(model, test_loader)  # , criterion)
    tester.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test mô hình phân loại ảnh")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Đường dẫn tới file cấu hình YAML",
    )
    args = parser.parse_args()

    # Gọi hàm train
    test_module1(config_path=args.config_path)
