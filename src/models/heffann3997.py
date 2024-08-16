import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from src.models.efficient_net import H0_EfficientNetB0
from src.models.ann import H39_97_ANN


class Heffann3997(nn.Module):
    def __init__(self):
        super(Heffann3997, self).__init__()
        self.efficient_net = H0_EfficientNetB0()
        self.ann = H39_97_ANN()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transforms = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )

    def load_weight(self, module_1_path: str, module_2_path: str):
        self.efficient_net.load_state_dict(
            torch.load(module_1_path, map_location=self.device)
        )
        self.ann.load_state_dict(torch.load(module_2_path, map_location=self.device))

    def eval_mode(self):
        self.eval()
        # self.efficient_net.eval()
        # self.ann.eval()

    def forward(self, x):
        """x is tensor of shape (batch_size, 3, 224, 224)"""
        if isinstance(x, torch.Tensor) and x.size(0) % 13 != 0:
            raise ValueError("batch_size must be multiple of 13")
        elif isinstance(x, list) and len(x) % 13 != 0:
            raise ValueError("batch_size must be multiple of 13")
        else:
            raise ValueError("x must be a tensor or a list of tensors")
        # print(x.shape)
        x = self.efficient_net(x)  # shape (batch_size, 3)
        # print(x.shape)
        x = x.view(-1, 39)  # shape (batch_size // 13, 13 * 3)
        # print(x.shape)
        x = self.ann(x)  # shape (batch_size // 13, 3)
        # print(x.shape)
        return x

    def predict_from_path(self, imgs_path):
        inputs = []
        for img_path in imgs_path:
            img = Image.open(img_path).convert("RGB").resize((1024, 768))
            crops = self._crop_image(img)
            crops = [transforms.ToTensor()(crop) for crop in crops]
            img = transforms.Resize((224, 224))(img)
            img = transforms.ToTensor()(img)
            x = torch.stack([img] + crops)
            inputs.append(x)
        inputs = torch.stack(inputs)
        inputs = inputs.to(self.device)
        inputs = inputs.view(-1, 3, 224, 224)
        outputs = self.forward(inputs)
        return outputs

    def predict_from_PIL(self, imgs_PIL):
        """Make sure that imgs is a list of PIL images and img is RGB"""
        inputs = []
        for img in imgs_PIL:
            crops = self._crop_image(img.resize((1024, 768)))
            crops = [transforms.ToTensor()(crop) for crop in crops]
            img = transforms.Resize((224, 224))(img)
            img = transforms.ToTensor()(img)
            x = torch.stack([img] + crops)
            inputs.append(x)
        inputs = torch.stack(inputs)
        inputs = inputs.to(self.device)
        outputs = self.forward(inputs)
        return outputs

    def _crop_image(self, img):
        img_crops = []
        width, height = img.size
        crop_width = 256
        crop_height = 256
        for i in range(3):  # 3 rows
            for j in range(4):  # 4 columns
                left = j * crop_width
                top = i * crop_height
                right = min(left + crop_width, width)
                bottom = min(top + crop_height, height)
                crop = img.crop((left, top, right, bottom)).resize((224, 224))
                img_crops.append(crop)
        return img_crops
