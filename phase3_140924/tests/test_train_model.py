import sys
import os

# Thêm thư mục gốc của dự án vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from src.training.train_model import TrainClassificationModel


# 1. Dữ liệu giả
class SimpleDataset(Dataset):
    def __init__(self, size=50):
        self.data = torch.rand(size, 3)
        self.labels = torch.randint(0, 3, (size,))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 2. Mô hình đơn giản
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 6)
        self.fc2 = nn.Linear(6, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 3. Tạo dữ liệu và mô hình
train_dataset = SimpleDataset()
val_dataset = SimpleDataset()
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = None  # Không sử dụng scheduler trong ví dụ này

# 4. Khởi tạo lớp TrainClassificationModel và chạy thử
config = {
    "trainer": {
        "model_type": "simple_nn",
        "num_epochs": 3,
        "patience": 1,
        "device": "cpu",
    }
}

# Tạo instance của lớp TrainClassificationModel và gọi phương thức train
trainer = TrainClassificationModel(config_path=None, **config)
trainer.setup(model, train_loader, val_loader, criterion, optimizer, scheduler)
trainer.train()

print("Test hoàn tất.")
