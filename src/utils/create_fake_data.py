# src/utils/create_fake_data.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class FakeDataset(Dataset):
    def __init__(self, num_samples, num_classes, image_size):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.images = np.random.rand(num_samples, *image_size).astype(np.float32)
        self.labels = np.random.randint(0, num_classes, num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx])
        label = torch.tensor(self.labels[idx])
        return image, label


def create_fake_dataloaders(
    batch_size=32, num_samples=1000, num_classes=3, image_size=(3, 224, 224)
):
    dataset = FakeDataset(num_samples, num_classes, image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, dataloader
