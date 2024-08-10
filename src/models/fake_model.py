# src/models/fake_model.py
import torch
import torch.nn as nn

class FakeModel(nn.Module):
    def __init__(self, num_classes):
        super(FakeModel, self).__init__()
        self.fc = nn.Linear(3 * 224 * 224, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.fc(x)
