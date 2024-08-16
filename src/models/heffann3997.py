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
        self.transforms = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.efficient_net.to(self.device)
        self.ann.to(self.device)
        self.eval()
        self.hook = None

    def load_weight(self, module_1_path: str, module_2_path: str):
        self.efficient_net.load_state_dict(
            torch.load(module_1_path, map_location=self.device)
        )
        self.ann.load_state_dict(torch.load(module_2_path, map_location=self.device))

    def forward(self, x, grad_cam=False):
        """x is either a path to an image or a PIL Image object"""
        if isinstance(x, str):
            x = Image.open(x).resize((1024, 768)).convert("RGB")
        elif isinstance(x, Image.Image):
            x = x.resize((1024, 768))
        else:
            raise ValueError("x must be a path to an image or a PIL Image object")

        # Step 2. Crop 12 patches 256x256 images from the image
        patches = []
        for i in range(3):
            for j in range(4):
                patch = x.crop((i * 256, j * 256, (i + 1) * 256, (j + 1) * 256))
                patches.append(patch)
        images = [x] + patches

        # Step 3. Use transforms to resize 224x224 and convert the patches to tensors
        images = [self.transforms(image).to(self.device) for image in images]

        # Step 4. Stack the tensors along the batch dimension
        x = torch.stack(images)

        x = self.efficient_net(x)

        # Step 5. Flatten and pass through ANN
        x = x.view(x.size(0), -1)  # Flatten the feature maps
        x = self.ann(x)

        return x

    @torch.no_grad()
    def predict(self, x):
        x = self.forward(x)
        return torch.argmax(x, dim=1).item()
