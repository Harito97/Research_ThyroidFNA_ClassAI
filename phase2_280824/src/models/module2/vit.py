import torch
import torch.nn as nn
import torch.nn.functional as F


class H13_63_ViT(nn.Module):
    def __init__(
        self,
        num_classes=3,
        dim=6,
        depth=3,
        heads=2,
        mlp_dim=12,
        dropout=0.1,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, 13 + 1, dim))
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                batch_first=True,  # Đặt batch_first=True
            ),
            num_layers=depth,
        )
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):  # x shape (2, 13, 3) = (batch_size, num_patches, dim)
        # print(f"self.cls_token: {self.cls_token}")
        # print(f"self.pos_embedding: {self.pos_embedding}")
        # print(f"x: {x}")
        # print(x.shape)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        # print(cls_tokens.shape)

        one_hot = F.one_hot(torch.argmax(x, dim=2), num_classes=x.size(2)).float()
        # print(one_hot)
        x = torch.cat((one_hot, x), dim=2)
        # print(f"After concat vs one_hot: {x}")
        x = torch.cat((cls_tokens, x), dim=1)
        # print(f"After concat vs cls_token: {x}")
        # print(x.shape)
        x += self.pos_embedding
        # print(f"After add pos_embedding: {x}")
        # print(x.shape)
        x = self.transformer(x)
        # print(x.shape)
        x = self.head(x[:, 0])
        # print(x.shape)
        # print(x)
        return x


if __name__ == "__main__":
    # Ví dụ sử dụng
    model = ViT(num_classes=3, dim=6, depth=3, heads=2, mlp_dim=12)
    x = torch.randn(2, 13, 3)  # Một batch gồm 2 ảnh
    output = model(x)

    num_params = sum(p.numel() for p in model.parameters())
    print(num_params)
