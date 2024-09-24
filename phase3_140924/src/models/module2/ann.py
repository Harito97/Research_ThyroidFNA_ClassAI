import torch
import torch.nn as nn


class ANN_Models(nn.Module):
    def __init__(
        self,
        input_dim: int = 39,
        dim_hidden: list = [79],
        num_classes: int = 3,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        """
        Khởi tạo mô hình ANN với các lớp ẩn tùy chỉnh, hàm kích hoạt và Dropout.

        :param input_dim: Kích thước đầu vào
        :param dim_hidden: Danh sách các kích thước ẩn của từng lớp
        :param num_classes: Số lớp đầu ra (số class phân loại)
        :param activation: Tên hàm kích hoạt ('relu', 'tanh', 'sigmoid')
        :param dropout: Tỷ lệ Dropout (float từ 0.0 đến 1.0)
        """
        super(ANN_Models, self).__init__()

        # Khởi tạo danh sách các lớp
        layers = []
        prev_dim = input_dim

        # Map tên hàm kích hoạt thành các hàm tương ứng của PyTorch
        activation_functions = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }

        # Lấy hàm kích hoạt dựa trên tên
        activation_fn = activation_functions.get(activation.lower(), nn.ReLU())

        # Xây dựng các lớp fully connected và kích hoạt
        for hidden_dim in dim_hidden:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_fn)  # Hàm kích hoạt sau mỗi lớp Linear

            if dropout > 0:  # Nếu tỷ lệ Dropout lớn hơn 0, thêm lớp Dropout
                layers.append(nn.Dropout(p=dropout))

            prev_dim = hidden_dim

        # Lớp output (fully connected layer cuối cùng)
        layers.append(nn.Linear(prev_dim, num_classes))

        # Gộp các lớp lại bằng nn.Sequential
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def get_ann_model(
    input_dim: int = 39,
    dim_hidden: list = [79],
    num_classes: int = 3,
    activation: str = "relu",
    dropout: float = 0.0,
) -> ANN_Models:
    """
    Hàm tạo và trả về mô hình ANN với hàm kích hoạt và Dropout tùy chỉnh.

    :param input_dim: Kích thước đầu vào
    :param dim_hidden: Danh sách các kích thước ẩn của từng lớp
    :param num_classes: Số lượng lớp đầu ra
    :param activation: Tên hàm kích hoạt ('relu', 'tanh', 'sigmoid')
    :param dropout: Tỷ lệ Dropout (float từ 0.0 đến 1.0)
    :return: Một instance của lớp ANN_Models
    """
    return ANN_Models(
        input_dim=input_dim,
        dim_hidden=dim_hidden,
        num_classes=num_classes,
        activation=activation,
        dropout=dropout,
    )


# Ví dụ sử dụng:
# model = get_ann_model(input_dim=39, dim_hidden=[79, 50], num_classes=3, activation='tanh', dropout=0.5)
# output = model(torch.randn(10, 39))  # Batch size 10, input size 39
