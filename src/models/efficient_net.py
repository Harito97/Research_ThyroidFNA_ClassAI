from torch import flatten
from torch.nn import Module, Sequential, Linear, ReLU, Dropout
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class H97_EfficientNetB0(Module):
    def __init__(self, num_classes: int = 3, retrain_whole_net: bool = False):
        super(H97_EfficientNetB0, self).__init__()
        # Load a pretrained EfficientNet model
        efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        # Remove the last fully connected layer
        self.feature_extractor = Sequential(*list(efficientnet.children())[:-1])
        if not retrain_whole_net:
            # Freeze the parameters in the feature extractor to not update during training
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # The input size for the first fully connected layer based on the output of EfficientNet
        # EfficientNet typically returns a tensor [batch_size, 1280, 1, 1] after the last pooling layer
        self.fc1 = Linear(1280, 9)
        self.fc2 = Linear(9, 7)
        self.fc3 = Linear(7, num_classes)
        self.dropout = Dropout(0.1)

        # Hook for CAM
        # self.features = None

    def forward(self, x):
        # Feature extraction
        x = self.feature_extractor(x)
        # self.features = x

        # Flatten the tensor from [batch_size, 1280, 1, 1] to [batch_size, 1280] to match the fully connected layer
        x = flatten(x, 1)

        # Pass through the dense network
        x = self.fc1(x)
        x = ReLU()(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = ReLU()(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return x
