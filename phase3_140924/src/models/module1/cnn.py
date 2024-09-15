import torch
import torch.nn as nn
import torchvision.models as models


def get_cnn_model(name, num_classes=3):
    if name == "vgg16":
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)
    elif name == "vgg19":
        model = models.vgg19(pretrained=True)
        model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)
    elif name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
    elif name == "resnet152":
        model = models.resnet152(pretrained=True)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
    elif name == "densenet121":
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(
            in_features=model.classifier.in_features, out_features=num_classes
        )
    elif name == "densenet201":
        model = models.densenet201(pretrained=True)
        model.classifier = nn.Linear(
            in_features=model.classifier.in_features, out_features=num_classes
        )
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(
            in_features=model.classifier[1].in_features, out_features=num_classes
        )
    elif name == "efficientnet_b7":
        model = models.efficientnet_b7(pretrained=True)
        model.classifier[1] = nn.Linear(
            in_features=model.classifier[1].in_features, out_features=num_classes
        )
    elif name == "mobilenet_v1":
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(
            in_features=model.classifier[1].in_features, out_features=num_classes
        )
    elif name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(pretrained=True)
        model.classifier[3] = nn.Linear(
            in_features=model.classifier[3].in_features, out_features=num_classes
        )
    else:
        raise ValueError(f"Model name {name} is not recognized.")

    return model
