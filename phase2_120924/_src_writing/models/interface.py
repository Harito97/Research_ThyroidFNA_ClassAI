from abc import ABC, abstractmethod
import torch.nn as nn

class FeatureExtractor(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def evaluate(self, data_loader):
        pass

    @abstractmethod
    def get_cnn_feature(self, data_loader):
        pass

    @abstractmethod
    def get_last_feature(self, data_loader):
        pass

# class MyModel(nn.Module, ClassificationModel):
#     def __init__(self, num_classes):
#         super(MyModel, self).__init__()
#         # Khởi tạo các lớp con của mạng
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         # ...

#     def forward(self, x):
#         # Thực hiện tính toán tiến tới
#         x = self.conv1(x)
#         # ...
#         return x

#     def evaluate(self, data_loader):
#         # Đánh giá mô hình
#         # ...
#         ...

class Classifier(ABC):
    @abstractmethod
    def train(self, data_loader):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def evaluate(self, data_loader):
        pass