import sys
import os

# Add root folder to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import MagicMock, patch
import torch
from src.utils.build_model.image_classification import TrainImageClassificationModel

@pytest.fixture
def setup_train_image_classification_model():
    model = MagicMock()
    train_loader = MagicMock()
    val_loader = MagicMock()
    criterion = MagicMock()
    optimizer = MagicMock()
    scheduler = MagicMock()
    num_epochs = 2
    patience = 1
    device = torch.device('cpu')
    save_path = 'results/test_image_classification_file/fake_model.pth'
    experiment_yaml_config = {
        "logging": {
            "project_name": "TestProject",
            "run_name": "TestRun"
        }
    }

    train_model = TrainImageClassificationModel(
        experiment_yaml_config=experiment_yaml_config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        patience=patience,
        device=device,
        save_path=save_path
    )

    return train_model, model, train_loader, val_loader, criterion, optimizer, scheduler

@patch('wandb.init')
@patch('wandb.log')
@patch('wandb.finish')
@patch('torch.save')
def test_train_model_init(mock_save, mock_finish, mock_wandb_log, mock_wandb_init, setup_train_image_classification_model):
    train_model, model, train_loader, val_loader, criterion, optimizer, scheduler = setup_train_image_classification_model

    assert train_model.model == model
    assert train_model.train_loader == train_loader
    assert train_model.val_loader == val_loader
    assert train_model.criterion == criterion
    assert train_model.optimizer == optimizer
    assert train_model.scheduler == scheduler
    assert train_model.num_epochs == 2
    assert train_model.patience == 1
    assert train_model.device == torch.device('cpu')
    assert train_model.save_path == 'results/test_image_classification_file/fake_model.pth'

    mock_wandb_init.assert_called_once_with(
        project="test_project",
        config={
            "epochs": 2,
            "learning_rate": optimizer.defaults["lr"],
            "batch_size": train_loader.batch_size,
        },
        name="test_run"
    )

@patch('wandb.log')
@patch('wandb.finish')
@patch('torch.save')
@patch('torch.no_grad')
def test_train_method(mock_no_grad, mock_save, mock_wandb_finish, mock_wandb_log, setup_train_image_classification_model):
    train_model, model, train_loader, val_loader, criterion, optimizer, scheduler = setup_train_image_classification_model

    model.train = MagicMock()
    model.eval = MagicMock()
    model.to = MagicMock(return_value=model)
    train_loader.__iter__ = MagicMock(return_value=iter([(torch.randn(1, 3, 224, 224), torch.tensor([0]))]))
    val_loader.__iter__ = MagicMock(return_value=iter([(torch.randn(1, 3, 224, 224), torch.tensor([0]))]))
    criterion.return_value = torch.tensor(0.0)
    optimizer.zero_grad = MagicMock()
    optimizer.step = MagicMock()
    scheduler.step = MagicMock()
    model.state_dict = MagicMock(return_value={})

    train_model.train()

    mock_wandb_log.assert_any_call({"train_loss": 0.0, "train_accuracy": 1.0})
    mock_wandb_log.assert_any_call({"valid_loss": 0.0, "valid_accuracy": 1.0})

    mock_save.assert_called_once_with(model.state_dict(), 'results/test_image_classification_file/fake_model.pth')
    mock_wandb_finish.assert_called_once()

if __name__ == "__main__":
    pytest.main()
