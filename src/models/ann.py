import torch
import torch.nn as nn

class H39_97_ANN(nn.Module):
    def __init__(self, input_dim=39, output_dim=3):
        super(H39_97_ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 97)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(97, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x