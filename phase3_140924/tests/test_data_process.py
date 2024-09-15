import unittest
import tempfile
import shutil
import os
from PIL import Image
from src.data.process import split_dataset, augment_images


class TestProcessFunctions(unittest.TestCase):

    def setUp(self):
        # Khởi tạo các đường dẫn dữ liệu
        self.raw_dir = "./tests/temp_data_for_tests/raw"
        self.processed_dir = "./tests/temp_data_for_tests/processed"
        self.augmented_dir = "./tests/temp_data_for_tests/augmented"
        self.model_path = "./results/final_weights/cluster_detect.pt"

        # Đảm bảo thư mục tồn tại
        if not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir)
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        if not os.path.exists(self.augmented_dir):
            os.makedirs(self.augmented_dir)

    def test_1_split_dataset(self):
        # Test hàm split_dataset
        try:
            train_dir, valid_dir, test_dir = split_dataset(
                self.raw_dir, self.processed_dir, 0.7, 0.2
            )
            self.assertTrue(os.path.exists(train_dir))
            self.assertTrue(os.path.exists(valid_dir))
            self.assertTrue(os.path.exists(test_dir))
            print(f"train_dir: {train_dir}")
            print(f"valid_dir: {valid_dir}")
            print(f"test_dir: {test_dir}")

            # Kiểm tra xem các hình ảnh có tồn tại không
            for subdir in [train_dir, valid_dir, test_dir]:
                self.assertTrue(len(os.listdir(subdir)) > 0)
                for img in os.listdir(subdir):
                    img_path = os.path.join(subdir, img)
                    with Image.open(img_path) as im:
                        self.assertEqual(im.format, "JPEG")  # Đảm bảo định dạng JPEG
        except Exception as e:
            print(e)

    def __check_for_single_subdir(self, input_dir, output_dir):
        augmented_dir = augment_images(input_dir, output_dir, self.model_path)
        self.assertTrue(os.path.exists(augmented_dir))
        # Kiểm tra xem hình ảnh đã được augment chưa
        for subdir in os.listdir(augmented_dir):
            self.assertTrue(len(os.listdir(subdir)) > 0)
            for img in os.listdir(subdir):
                img_path = os.path.join(augmented_dir, subdir, img)
                with Image.open(img_path) as im:
                    self.assertEqual(im.format, "JPEG")

    def test_2_augment_images(self):
        # Test hàm augment_images
        try:
            for dataset_name in os.listdir(self.processed_dir):
                for subdir in os.listdir(
                    os.path.join(self.processed_dir, dataset_name)
                ):
                    if subdir not in ["train", "valid"]:
                        print(
                            f"dataset_name: {dataset_name} can't augment for {subdir} dataset"
                        )
                        continue
                    input_dir = os.path.join(self.processed_dir, dataset_name, subdir)
                    output_dir = self.augmented_dir
                    self.__check_for_single_subdir(input_dir, output_dir)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    # Mở file để lưu kết quả
    with open("./test/test_data_process_results.txt", "w") as f:
        # Khởi tạo test suite
        suite = unittest.TestLoader().loadTestsFromTestCase(TestProcessFunctions)
        # Chạy test và ghi kết quả vào file
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        runner.run(suite)
