import torch
from torchvision.models import vision_transformer as vit

def get_vit_model(name, num_classes):
    if name == 'vit_b_16':
        model = vit.vit_b_16(pretrained=True, num_classes=num_classes)
    elif name == 'vit_l_16':
        model = vit.vit_l_16(pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f"Model name {name} is not recognized.")
    
    return model