import torch
import torch.nn as nn

class H39_97_ANN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(H39_97_ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 97)
        # self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(97, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        # x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x