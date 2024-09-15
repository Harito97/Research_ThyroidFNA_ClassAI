import os
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from PIL import Image
from collections import defaultdict


def load_random_images(folder_path, num_images=100):
    image_files = []
    labels = []
    folder_count = defaultdict(int)

    # Duyệt qua tất cả các thư mục con
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(("png", "jpg", "jpeg", "bmp", "tiff")):
                file_path = os.path.join(root, file)

                if ".ipynb_checkpoints" in file_path:
                    continue

                # # Lấy nhãn từ phần tên file trước dấu '/'
                # label = file_path.split("/")[-2]
                # Lấy nhãn từ tên thư mục cha của file ảnh
                label = os.path.basename(os.path.dirname(file_path))

                # Thêm file ảnh và nhãn vào danh sách
                image_files.append(file_path)
                labels.append(label)

                folder_count[
                    os.path.dirname(file_path)
                ] += 1  # Tăng số lượng ảnh tương ứng với folder

    # In ra số lượng ảnh trong mỗi thư mục
    print("Số lượng ảnh trong mỗi folder:")
    for folder, count in folder_count.items():
        print(f"{folder}: {count} ảnh")

    # Shuffle và lấy ngẫu nhiên num_images ảnh, nếu không đủ thì lấy số lượng ảnh tối đa
    combined = list(zip(image_files, labels))
    random.shuffle(combined)
    selected_files, selected_labels = zip(*combined[: min(num_images, len(combined))])

    # Đọc các ảnh vào danh sách
    images = [Image.open(f).resize((64, 64)) for f in selected_files]
    return images, selected_labels


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


def plot_interactive_3d(data, labels):
    # Lấy danh sách các nhãn duy nhất
    unique_labels = list(set(labels))

    # Tạo một dictionary để ánh xạ nhãn thành các giá trị số (cho phép vẽ màu khác nhau)
    label_to_color = {label: idx for idx, label in enumerate(unique_labels)}
    colors = [label_to_color[label] for label in labels]

    # Vẽ biểu đồ 3D tương tác
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        data[:, 0], data[:, 1], data[:, 2], c=colors, cmap="viridis", marker="o"
    )

    # Thêm thanh màu để biết nhãn tương ứng với màu gì
    legend1 = ax.legend(*scatter.legend_elements(), title="Labels")
    ax.add_artist(legend1)

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    plt.show()


def main(folder_path: str = "/path/to/your/folder"):
    # Load ngẫu nhiên 100 ảnh từ thư mục và các nhãn tương ứng
    images, labels = load_random_images(folder_path, num_images=100)

    # Hiển thị các ảnh đã chọn
    plot_images(images)

    # Giảm chiều dữ liệu ảnh xuống 3 chiều
    reduced_data = reduce_dimensionality(images)

    # Vẽ biểu đồ 3D tương tác với nhãn
    plot_interactive_3d(reduced_data, labels)


if __name__ == "__main__":
    folder_path = "/path/to/your/folder"
    main(folder_path=folder_path)
