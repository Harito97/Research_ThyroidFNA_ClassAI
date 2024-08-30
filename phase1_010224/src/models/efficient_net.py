# src/models/efficient_net.py

# from torch import flatten
# from torch.nn import Module, Sequential, Linear, ReLU, Dropout
# from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class H0_EfficientNetB0(nn.Module):
    def __init__(
        self,
        config=None,
        num_classes=3,
        train_entire_network=False,
    ):
        super(H0_EfficientNetB0, self).__init__()

        # Load configuration
        if config is not None:
            num_classes = len(config["data_para"]["classes"])
            train_entire_network = config["train_valid_para"]["train_entire_network"]

        # Initialize model with pretrained weights
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        # Replace the final classification layer
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

        # Freeze all layers except the final classifier layer if not retraining the whole network
        if not train_entire_network:
            # Freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False
            # Unfreeze only the parameters of the final classifier layer
            for param in self.model.classifier[1].parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)
