# src/data/dataset/image_dataset.py
import os
import glob
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset

class MultiImageFolderDataset(Dataset):
    def __init__(self, config, root_dirs, transform=None):
        """
        Use: create a dataset from multiple folders containing images 
        """
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        for idx, class_name in enumerate(config["data_para"]["classes"]):
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name

        for root_dir in root_dirs:
            if not os.path.isdir(root_dir):
                raise ValueError(f"{root_dir} is not a exist directory")
            for label in config["data_para"]["classes"]:
                # Tìm tất cả các ảnh trong thư mục con
                image_paths = glob.glob(os.path.join(root_dir, label, '*.jpg'))  # Thay đổi định dạng file nếu cần
                self.image_paths.extend(image_paths)
                self.labels.extend([self.class_to_idx[label]] * len(image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_labels(self):
        return self.labels

if __name__ == "__main__":
    # Eg to use the dataset
    from torchvision import transforms
    from torch.utils.data import DataLoader

    # Các thư mục chứa dữ liệu
    root_dirs = ['path/to/folder_A', 'path/to/folder_B']

    # Các biến đổi dữ liệu
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Khởi tạo dataset và dataloader
    dataset = MultiFolderDataset(root_dirs=root_dirs, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # Sử dụng dataloader trong quá trình huấn luyện
    for images, labels in dataloader:
        # Huấn luyện mô hình
        pass
