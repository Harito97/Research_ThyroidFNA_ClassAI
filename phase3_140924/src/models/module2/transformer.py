import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerModel(nn.Module):
    def __init__(
        self,
        num_patches: int = 13,
        num_classes: int = 3,
        dim: int = 9,
        depth: int = 3,
        heads: int = 3,
        mlp_dim: int = 12,
        hh_dim: int = 7,
        dropout: float = 0.3,
        activation: str = "relu",
    ):
        """
        Khởi tạo mô hình Transformer với các tham số tùy chỉnh.

        :param num_patches: Số lượng patches (phân đoạn) đầu vào
        :param num_classes: Số lớp đầu ra
        :param dim: Kích thước ẩn cho Transformer
        :param depth: Số lớp encoder trong Transformer
        :param heads: Số lượng heads trong cơ chế attention
        :param mlp_dim: Kích thước của MLP (feedforward network)
        :param hh_dim: Kích thước của hidden layer trong head classifier (fully connected)
        :param dropout: Tỷ lệ dropout
        :param activation: Hàm kích hoạt sử dụng trong encoder layer
        """
        super(TransformerModel, self).__init__()

        # Token đặc biệt [CLS] để tổng hợp thông tin
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        # Positional embedding cho các patch và cls_token
        # self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        # self.pos_embedding = nn.Parameter(
        #     torch.zeros(1, num_patches, dim)
        # )  # as use first patch (image level embedding) as cls token
        # self.post_embedding is a matrix of size (num_classes, dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, num_classes, dim))

        # Xây dựng lớp TransformerEncoder với các tham số tùy chỉnh
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,  # Bố trí batch ở vị trí đầu
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=depth
        )

        # Lớp fully connected head để dự đoán
        self.head = nn.Sequential(
            nn.Linear(dim, hh_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hh_dim, num_classes),
        )

        # Số lượng tham số của mô hình
        self.num_paras = self.__get_num_paras()

    def forward(self, x):
        """
        Forward pass của mô hình Transformer.

        :param x: Tensor đầu vào có shape (batch_size, num_patches, dim). Eg: (-1, 13, 3)
        :return: Dự đoán của mô hình
        """
        batch_size = x.size(0)

        # Thêm chiều cho x để nhân với positional embedding
        x = x.unsqueeze(2)  # (batch_size, num_patches, 1, dim)

        # Nhân ma trận sử dụng bmm
        # pos_embedding cần có shape (1, 13, 3, 9) để mở rộng cho batch

        x = torch.bmm(x, self.pos_embedding)  # (batch_size, num_patches, 1, 9)

        # Loại bỏ chiều không cần thiết
        x = x.squeeze(2)  # (batch_size, num_patches, 9)

        # Transformer Encoder
        x = self.transformer(x)  # (batch_size, num_patches, 9)

        # Lấy đầu ra từ cls_token (vị trí đầu tiên)
        x = x[:, 0]  # (batch_size, 9)

        x = self.head(x)  # Kích thước bây giờ là (batch_size, 3)

        return x

    def __get_num_paras(self):
        return sum(p.numel() for p in self.parameters())


def get_transformer_model(
    num_patches: int = 13,
    num_classes: int = 3,
    dim: int = 9,
    depth: int = 3,
    heads: int = 3,
    mlp_dim: int = 12,
    dropout: float = 0.3,
    activation: str = "relu",
    add_pos_embedding: bool = True,
) -> TransformerModel:
    """
    Hàm tạo và trả về mô hình Transformer với các tham số tùy chỉnh.

    :param num_patches: Số lượng patches
    :param num_classes: Số lớp đầu ra
    :param dim: Kích thước ẩn cho Transformer
    :param depth: Số lớp encoder trong Transformer
    :param heads: Số lượng heads trong cơ chế attention
    :param mlp_dim: Kích thước của MLP
    :param dropout: Tỷ lệ dropout
    :param activation: Hàm kích hoạt ('relu', 'gelu', 'tanh')
    :param add_pos_embedding: Có thêm positional embedding hay không
    :return: Một instance của lớp TransformerModel
    """
    return TransformerModel(
        num_patches=num_patches,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        activation=activation,
        add_pos_embedding=add_pos_embedding,
    )


if __name__ == "__main__":
    # Ví dụ sử dụng:
    model = get_transformer_model(
        num_patches=13,
        num_classes=3,
        dim=9,
        depth=3,
        heads=3,
        mlp_dim=12,
        dropout=0.3,
        activation="relu",
        add_pos_embedding=True,
    )
    x = torch.randn(2, 13, 3)  # Một batch gồm 2 ảnh, mỗi ảnh có 13 patches và dim=6
    output = model(x)

    num_params = model.num_paras
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {num_params}")
