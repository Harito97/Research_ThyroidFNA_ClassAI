# src/models/efficient_net.py

# from torch import flatten
# from torch.nn import Module, Sequential, Linear, ReLU, Dropout
# from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights


class H97_9_7_EfficientNetB0(nn.Module):
    def __init__(
        self,
        experiment_yaml_config=None,
        num_classes=3,
        retrain_whole_net=False,
        dropout_rate=0,
    ):
        super(H97_9_7_EfficientNetB0, self).__init__()

        # Load configuration
        if experiment_yaml_config is not None:
            num_classes = len(experiment_yaml_config["classes"])
            retrain_whole_net = experiment_yaml_config["training"]["retrain_whole_net"]
            dropout_rate = experiment_yaml_config["training"]["dropout_rate"]

        # Load a pretrained EfficientNet model
        efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        # Remove the last fully connected layer
        self.feature_extractor = nn.Sequential(*list(efficientnet.children())[:-1])
        if not retrain_whole_net:
            # Freeze the parameters in the feature extractor to not update during training
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        self.fc1 = nn.Linear(1280, 97)
        self.fc2 = nn.Linear(97, 9)
        self.fc3 = nn.Linear(9, 7)
        self.fc4 = nn.Linear(7, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

        # Hook for CAM
        # self.features = None

    def forward(self, x):
        # Feature extraction
        x = self.feature_extractor(x)
        # self.features = x

        # Flatten the tensor from [batch_size, 1280, 1, 1] to [batch_size, 1280]
        # to match the fully connected layer
        x = torch.flatten(x, 1)

        # Pass through the dense network
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x


class H0_EfficientNetB0(nn.Module):
    def __init__(
        self,
        experiment_yaml_config=None,
        num_classes=3,
        retrain_whole_net=False,
        dropout_rate=0,
    ):
        super(H0_EfficientNetB0, self).__init__()

        # Load configuration
        if experiment_yaml_config is not None:
            num_classes = len(experiment_yaml_config["classes"])
            retrain_whole_net = experiment_yaml_config["training"]["retrain_whole_net"]
            dropout_rate = experiment_yaml_config["training"]["dropout_rate"]

        # Initialize model with pretrained weights
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        # Replace the final classification layer
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

        # Freeze all layers except the final classifier layer if not retraining the whole network
        if not retrain_whole_net:
            # Freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False
            # Unfreeze only the parameters of the final classifier layer
            for param in self.model.classifier[1].parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)

class H0_EfficientNetB7(nn.Module):
    def __init__(
        self,
        experiment_yaml_config=None,
        num_classes=3,
        retrain_whole_net=False,
        dropout_rate=0,
    ):
        super(H0_EfficientNetB7, self).__init__()

        # Load configuration
        if experiment_yaml_config is not None:
            num_classes = len(experiment_yaml_config["classes"])
            retrain_whole_net = experiment_yaml_config["training"]["retrain_whole_net"]
            dropout_rate = experiment_yaml_config["training"]["dropout_rate"]

        # Initialize model with pretrained weights
        self.model = efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT)

        # Replace the final classification layer
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

        # Freeze all layers except the final classifier layer if not retraining the whole network
        if not retrain_whole_net:
            # Freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False
            # Unfreeze only the parameters of the final classifier layer
            for param in self.model.classifier[1].parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)