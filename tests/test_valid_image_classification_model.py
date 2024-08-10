# src/tests/test_valid_image_classification_model.py
import sys
import os

# Add root folder to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# import torch
# from src.utils.create_fake_data import create_fake_dataloaders
# from src.models.fake_model import FakeModel
# from src.utils.build_model.image_classification import ValidImageClassificationModel


# def test_valid_image_classification_model():
#     # Parameters
#     num_classes = 3
#     batch_size = 32
#     image_size = (3, 224, 224)
#     num_samples = 1000
#     experiment_yaml_config = {
#         "logging": {
#             "save_path": "results/test_image_classification_file",
#             "project_name": "TestProject",
#             "run_name": "TestRun",
#         },
#         "training": {
#             "save_path": "results/test_image_classification_file/fake_model.pth"
#         },
#     }

#     # Create fake dataloaders
#     val_loader, _ = create_fake_dataloaders(
#         batch_size=batch_size,
#         num_samples=num_samples,
#         num_classes=num_classes,
#         image_size=image_size,
#     )

#     # Create fake model and save it
#     model = FakeModel(num_classes=num_classes)
#     # Check if path exists
#     if not os.path.exists(experiment_yaml_config["training"]["save_path"]):
#         os.makedirs(
#             os.path.dirname(experiment_yaml_config["training"]["save_path"]),
#             exist_ok=True,
#         )
#     torch.save(model.state_dict(), experiment_yaml_config["training"]["save_path"])

#     # Initialize and run validation
#     validator = ValidImageClassificationModel(
#         experiment_yaml_config=experiment_yaml_config,
#         model=model,
#         val_loader=val_loader,
#     )
#     validator.evaluate()


# if __name__ == "__main__":
#     test_valid_image_classification_model()

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from src.utils.build_model.image_classification import ValidImageClassificationModel

@pytest.fixture
def setup_valid_image_classification_model():
    model = MagicMock()
    val_loader = MagicMock()
    experiment_yaml_config = {
        "logging": {
            "project_name": "test_project",
            "run_name": "test_run",
            "save_path": "results/test_image_classification_file"
        },
        "training": {
            "save_path": "results/test_image_classification_file/fake_model.pth"
        }
    }

    valid_model = ValidImageClassificationModel(
        experiment_yaml_config=experiment_yaml_config,
        model=model,
        val_loader=val_loader
    )

    return valid_model, model, val_loader

@patch('torch.load')
@patch('wandb.init')
@patch('wandb.log')
@patch('wandb.finish')
@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.close')
def test_valid_model_init(mock_close, mock_savefig, mock_wandb_finish, mock_wandb_log, mock_wandb_init, setup_valid_image_classification_model):
    valid_model, model, val_loader = setup_valid_image_classification_model

    assert valid_model.model == model
    assert valid_model.val_loader == val_loader
    assert valid_model.device == torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert valid_model.save_path == "results/test_image_classification_file"
    assert valid_model.model_best_path == "results/test_image_classification_file/best_model.pth"

    mock_wandb_init.assert_called_once_with(
        project="TestProject",
        config={
            "save_path": "results/test_image_classification_file",
        },
        name="TestRunValidation",
    )

@patch('wandb.log')
@patch('wandb.finish')
@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.close')
@patch('sklearn.metrics.confusion_matrix')
@patch('sklearn.metrics.roc_curve')
@patch('sklearn.metrics.auc')
@patch('sklearn.metrics.classification_report')
def test_evaluate_method(
    mock_classification_report, mock_auc, mock_roc_curve, mock_confusion_matrix,
    mock_close, mock_savefig, mock_wandb_finish, mock_wandb_log, setup_valid_image_classification_model
):
    valid_model, model, val_loader = setup_valid_image_classification_model

    model.to = MagicMock(return_value=model)
    model.eval = MagicMock()
    val_loader.__iter__ = MagicMock(return_value=iter([(torch.randn(1, 3, 224, 224), torch.tensor([0]))]))

    mock_classification_report.return_value = {'accuracy': 1.0}
    mock_auc.return_value = 1.0
    mock_confusion_matrix.return_value = np.array([[1, 0], [0, 1]])
    mock_roc_curve.return_value = (np.array([0, 1]), np.array([0, 1]), np.array([0, 1]))

    valid_model.evaluate()

    mock_wandb_log.assert_any_call({
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1_score": 1.0,
        "classification_report": MagicMock(),
        "roc_curve": MagicMock(),
        "confusion_matrix": MagicMock(),
    })

    mock_wandb_finish.assert_called_once()
    mock_savefig.assert_any_call("results/test_image_classification_file/classification_report.png")
    mock_savefig.assert_any_call("results/test_image_classification_file/roc_curve.png")
    mock_savefig.assert_any_call("results/test_image_classification_file/confusion_matrix.png")

if __name__ == "__main__":
    pytest.main()

