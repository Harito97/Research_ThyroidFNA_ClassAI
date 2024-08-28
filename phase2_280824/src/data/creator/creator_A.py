# src/data/creator/creator_A.py
import os
import shutil
import random
import time


def split_dataset(path, train_ratio=0.7, valid_ratio=0.15):
    # Kiểm tra đường dẫn tồn tại
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist.")

    # Tạo các thư mục con cho train, valid và test
    A_set_dir = f"data/{int(time.time())}_A_set"
    # if config["creator"]["A_set"] and not os.path.exists(A_set_dir):
    #     os.makedirs(A_set_dir, exist_ok=True)

    train_dir = os.path.join(A_set_dir, "train")
    valid_dir = os.path.join(A_set_dir, "valid")
    test_dir = os.path.join(A_set_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Lấy danh sách các thư mục nhãn
    label_dirs = [
        d
        for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d)) and d.startswith("B")
    ]
    # label_dirs = ['B2', 'B5', 'B6']

    # Print dataset information
    print("Dataset Information:")
    print(f"Total number of labels: {len(label_dirs)}")
    print(f"Labels: {label_dirs}")
    total_train, total_valid, total_test = 0, 0, 0

    for label_dir in label_dirs:
        label_path = os.path.join(path, label_dir)

        # Lấy tất cả các ảnh trong thư mục nhãn
        images = [
            f
            for f in os.listdir(label_path)
            if os.path.isfile(os.path.join(label_path, f))
        ]

        # Shuffle ảnh để đảm bảo việc phân chia ngẫu nhiên
        random.shuffle(images)

        # Tính toán số lượng ảnh cho mỗi tập
        num_images = len(images)
        num_train = int(train_ratio * num_images)
        num_valid = int(valid_ratio * num_images)
        num_test = num_images - num_train - num_valid

        total_train += num_train
        total_valid += num_valid
        total_test += num_test

        print(
            f"Label {label_dir}: {num_images} images, {num_train} train, {num_valid} valid, {num_test} test"
        )

        # Tạo các thư mục con cho nhãn trong train, valid và test
        os.makedirs(os.path.join(train_dir, label_dir), exist_ok=True)
        os.makedirs(os.path.join(valid_dir, label_dir), exist_ok=True)
        os.makedirs(os.path.join(test_dir, label_dir), exist_ok=True)

        # Di chuyển ảnh vào các thư mục tương ứng
        for i, image in enumerate(images):
            if i < num_train:
                dest_dir = os.path.join(train_dir, label_dir)
            elif i < num_train + num_valid:
                dest_dir = os.path.join(valid_dir, label_dir)
            else:
                dest_dir = os.path.join(test_dir, label_dir)

            # shutil.move(os.path.join(label_path, image), os.path.join(dest_dir, image))
            shutil.copy(os.path.join(label_path, image), os.path.join(dest_dir, image))

    print(f"Total: {total_train} train, {total_valid} valid, {total_test} test")
    print(
        f"Data has been split into train, valid, and test sets with ratios {train_ratio}, {valid_ratio}, and {1 - train_ratio - valid_ratio}."
    )
    print(f"Data saved to {A_set_dir}")
    return A_set_dir


def run(config):
    if not config["creator"]["A_set"]:
        print("Skipping data A creator")
        return
    start_time = time.time()
    print("Running data A creator")
    print(f"Path of raw dataset: {config['data']['path']}")
    print(f"Train ratio: {config['data']['train_ratio']}")
    print(f"Valid ratio: {config['data']['valid_ratio']}")
    print(
        f"Test ratio: {1 - config['data']['train_ratio'] - config['data']['valid_ratio']}"
    )
    A_set_dir = split_dataset(
        config["data"]["path"],
        config["data"]["train_ratio"],
        config["data"]["valid_ratio"],
    )
    print(f"Data A creator finished in {time.time() - start_time} seconds.")
    return A_set_dir


if __name__ == "__main__":
    # Sử dụng hàm
    path = "/path/to/your/dataset"
    split_dataset(path, train_ratio=0.7, valid_ratio=0.2)

    # or get the config from a file
    # config = {...}
    # run(config)
