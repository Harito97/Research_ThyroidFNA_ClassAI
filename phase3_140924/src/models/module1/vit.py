import torch
import torch.nn as nn
import timm


def get_vit_model(name, num_classes):
    if name == "vit_b_16":
        model = timm.create_model("vit_base_patch16_224", pretrained=True)
        model.head = nn.Linear(
            in_features=model.head.in_features, out_features=num_classes
        )
    elif name == "vit_l_16":
        model = timm.create_model("vit_large_patch16_224", pretrained=True)
        model.head = nn.Linear(
            in_features=model.head.in_features, out_features=num_classes
        )
    else:
        raise ValueError(f"Model name {name} is not recognized.")

    return model
