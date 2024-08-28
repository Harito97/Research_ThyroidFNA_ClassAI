import os
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from PIL import Image


def load_random_images(folder_path, num_images=100):
    image_files = []

    # Duyệt qua tất cả các thư mục con
    for root, _, files in os.walk(folder_path):
        # Lọc các file ảnh
        image_files_in_folder = [
            os.path.join(root, f)
            for f in files
            if f.endswith(("png", "jpg", "jpeg", "bmp", "tiff"))
        ]

        # In ra số lượng ảnh trong thư mục mức cuối cùng
        if len(image_files_in_folder) > 0:
            print(f"Found {len(image_files_in_folder)} images in {root}")

        # Thêm các file ảnh từ thư mục này vào danh sách tổng
        image_files.extend(image_files_in_folder)

    # Shuffle và lấy ngẫu nhiên num_images ảnh, nếu không đủ thì lấy số lượng ảnh tối đa
    random.shuffle(image_files)
    selected_files = image_files[: min(num_images, len(image_files))]

    # Đọc các ảnh vào danh sách
    images = [Image.open(f).resize((64, 64)) for f in selected_files]
    return images


def plot_images(images):
    # Vẽ các ảnh đã chọn ra màn hình
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i])
            ax.axis("off")
        else:
            ax.remove()
    plt.show()


def reduce_dimensionality(images):
    # Chuyển các ảnh thành vector
    image_vectors = np.array([np.array(img).flatten() for img in images])

    # Áp dụng PCA để giảm chiều xuống còn 50 chiều trước khi dùng t-SNE
    pca = PCA(n_components=50)
    reduced_data_pca = pca.fit_transform(image_vectors)

    # Áp dụng t-SNE để giảm chiều xuống còn 3
    tsne = TSNE(n_components=3, random_state=42)
    reduced_data_tsne = tsne.fit_transform(reduced_data_pca)

    return reduced_data_tsne


def plot_interactive_3d(data):
    # Vẽ biểu đồ 3D tương tác
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c="r", marker="o")

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    plt.show()


def main(folder_path: str = "/path/to/your/folder"):
    # Load ngẫu nhiên 100 ảnh từ thư mục
    images = load_random_images(folder_path, num_images=100)

    # Hiển thị các ảnh đã chọn
    plot_images(images)

    # Giảm chiều dữ liệu ảnh xuống 3 chiều
    reduced_data = reduce_dimensionality(images)

    # Vẽ biểu đồ 3D tương tác
    plot_interactive_3d(reduced_data)


if __name__ == "__main__":
    folder_path = "/path/to/your/folder"
    main(folder_path=folder_path)
