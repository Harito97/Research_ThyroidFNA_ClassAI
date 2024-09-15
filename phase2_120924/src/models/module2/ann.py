import torch
import torch.nn as nn

class ANN(nn.Module):
    def __init__(self, input_dim=39, output_dim=3, num_hidden=9, num_layers=3, dropout=0.5):
        super(ANN, self).__init__()
        self.start = nn.Linear(input_dim, num_hidden)
        self.hidden = nn.ModuleList([nn.Linear(num_hidden, num_hidden) for _ in range(num_layers - 1)])
        self.end = nn.Linear(num_hidden, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.relu(self.start(x))
        x = self.dropout(x)
        for layer in self.hidden:
            x = torch.relu(layer(x))
            x = self.dropout(x)
        x = self.end(x)
        return x