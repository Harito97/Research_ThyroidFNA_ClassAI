import torch
import torch.nn as nn
import torchvision.models as models

def get_cnn_model(name, num_classes=3):
    if name == 'vgg16':
        model = models.vgg16(pretrained=True, num_classes=num_classes)
    elif name == 'vgg19':
        model = models.vgg19(pretrained=True, num_classes=num_classes)
    elif name == 'resnet18':
        model = models.resnet18(pretrained=True, num_classes=num_classes)
    elif name == 'resnet152':
        model = models.resnet152(pretrained=True, num_classes=num_classes)
    elif name == 'densenet121':
        model = models.densenet121(pretrained=True, num_classes=num_classes)
    elif name == 'densenet201':
        model = models.densenet201(pretrained=True, num_classes=num_classes)
    elif name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True, num_classes=num_classes)
    elif name == 'efficientnet_b7':
        model = models.efficientnet_b7(pretrained=True, num_classes=num_classes)
    elif name == 'mobilenet_v1':
        model = models.mobilenet_v2(pretrained=True, num_classes=num_classes)
    elif name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f"Model name {name} is not recognized.")
    
    return model
