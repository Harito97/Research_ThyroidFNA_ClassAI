import os
import sys
import torch

# Add src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.process import PrepareDataTrainModule2

if __name__ == "__main__":
    # Prepare data for Module 2
    print("Preparing data for Module 2...")

    # Data directories
    root_data_dir_not_aug = "./data/processed/1726417351_70_15_15_42"
    root_data_dir_aug = (
        "./data/augmented"
    )
    train_not_aug = root_data_dir_not_aug + "/train"
    val_not_aug = root_data_dir_not_aug + "/val"
    test_not_aug = root_data_dir_not_aug + "/test"
    train_aug = root_data_dir_aug + "/augmented_train_1726417351_70_15_15_42"

    # Model directory
    model_path = "./logs/1726492356_efficientnet_b0"
    model_path += "/best_f1_model.pth"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Save directory
    save_dir = "./data/data_for_module2"
    os.makedirs(save_dir, exist_ok=True)

    # Prepare data
    prepare_data = PrepareDataTrainModule2(
        None,
        model_path=model_path,
        model_type="efficientnet_b0",
        device="cuda",
        num_classes=3,
    )

    # Prepare for train (not aug)
    prepare_data.set_data_dir(train_not_aug)
    prepare_data.process(path_save=os.path.join(save_dir, "train_not_aug.csv"))

    # Prepare for val (not aug)
    prepare_data.set_data_dir(val_not_aug)
    prepare_data.process(path_save=os.path.join(save_dir, "val_not_aug.csv"))

    # Prepare for test (not aug)
    prepare_data.set_data_dir(test_not_aug)
    prepare_data.process(path_save=os.path.join(save_dir, "test_not_aug.csv"))

    # Prepare for train (aug)
    prepare_data.set_data_dir(train_aug)
    prepare_data.process(path_save=os.path.join(save_dir, "train_aug.csv"))
