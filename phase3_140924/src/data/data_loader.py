# data_loader.py content
import os
import random
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class GroupedImageSampler(torch.utils.data.Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.groups = ["D", "C", "B", "A"]
        self.group_indices = {group: [] for group in self.groups}
        self.group_sizes = {group: 0 for group in self.groups}
        self._build_group_indices()

    def _build_group_indices(self):
        for idx, (image, label) in enumerate(self.dataset):
            for group in self.groups:
                if group in self.dataset.image_paths[idx]:
                    self.group_indices[group].append(idx)
                    self.group_sizes[group] += 1
                    break

    def __iter__(self):
        indices = []
        for group in self.groups:
            np.random.shuffle(self.group_indices[group])  # Shuffle within the group
            indices.extend(self.group_indices[group])
        return iter(indices)

    def __len__(self):
        return sum(len(self.group_indices[group]) for group in self.groups)


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images, organized in subdirectories by class.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = self._get_image_paths()
        self.image_paths = self._shuffle_within_groups(self.image_paths)
        self.label_dict = {"B2": 0, "B5": 1, "B6": 2}

    def _get_image_paths(self):
        """
        Collect image paths and sort them based on the specific naming convention.
        """
        image_paths = {"D": [], "C": [], "B": [], "A": [], "other": []}
        for label in sorted(os.listdir(self.root_dir)):
            label_dir = os.path.join(self.root_dir, label)
            if os.path.isdir(label_dir):
                for file_name in sorted(os.listdir(label_dir)):
                    if file_name.endswith((".jpg", ".jpeg", ".png")):
                        if "_D_patch_" in file_name:
                            image_paths["D"].append(os.path.join(label_dir, file_name))
                        elif "_C_crop_" in file_name:
                            image_paths["C"].append(os.path.join(label_dir, file_name))
                        elif "_B." in file_name:
                            image_paths["B"].append(os.path.join(label_dir, file_name))
                        elif "_A." in file_name:
                            image_paths["A"].append(os.path.join(label_dir, file_name))
                        else:
                            image_paths["other"].append(
                                os.path.join(label_dir, file_name)
                            )
        return image_paths

    def _shuffle_within_groups(self, image_paths):
        """
        Shuffle images within each group but keep group order intact.
        """
        shuffled_paths = []
        for group in ["D", "C", "B", "A", "other"]:
            random.shuffle(image_paths[group])
            shuffled_paths.extend(image_paths[group])
        return shuffled_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # Lấy tên lớp từ tên thư mục chứa ảnh
        label_name = os.path.basename(os.path.dirname(img_path))

        # Chuyển đổi tên lớp thành số lớp theo từ điển label_dict
        label = self.label_dict.get(
            label_name, -1
        )  # Sử dụng -1 nếu tên lớp không hợp lệ

        if self.transform:
            image = self.transform(image)

        return image, label


# Example usage of CustomImageDataset
def get_dataloader(
    root_dir, batch_size=32, shuffle=False, num_workers=4, transform=None
):
    dataset = CustomImageDataset(root_dir=root_dir, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return dataloader

# def get_augment_dataloader(
#     root_dir, batch_size=32, num_workers=4, transform=None
# ):
#     dataset = CustomImageDataset(root_dir=root_dir, transform=transform)
#     sampler = GroupedImageSampler(dataset)
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         sampler=sampler,
#         num_workers=num_workers
#     )
#     return dataloader

