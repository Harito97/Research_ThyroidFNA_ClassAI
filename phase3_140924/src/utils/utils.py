# utils.py content
import yaml


def load_config(config_path=None, **kwargs):
    """
    Đọc cấu hình từ file YAML hoặc từ các tham số keyword.

    Args:
        config_path (str, optional): Đường dẫn đến file cấu hình YAML. Nếu không cung cấp, cấu hình sẽ được lấy từ các tham số keyword.
        **kwargs: Các tham số keyword được sử dụng làm cấu hình nếu `config_path` không được cung cấp.

    Returns:
        dict: Cấu hình dưới dạng dictionary.
    """
    if config_path:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = kwargs
    return config
