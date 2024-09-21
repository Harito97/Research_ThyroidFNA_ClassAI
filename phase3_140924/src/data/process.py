# process.py content
# Used to process the data for the Thyroid FNA classification project.
import os
import sys

# Thêm thư mục gốc của dự án vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

####################################################################################################
# Split before pass module 1                                                                       #
####################################################################################################

import os
import random
import shutil
import time


def split_dataset(
    data_dir: str = "./data/raw",
    destination_dir: str = "./data/processed",
    train_ratio=0.7,
    valid_ratio=0.15,
    seed=42,
):
    """
    Use: Split the data into train, validation, and test sets.
    Input:
        - data_dir: Path to the images directory.
                    Eg:
                    data_dir = 'data' then
                    'data/B2/__images_here__',
                    'data/B5/__images_here__',
                    'data/B6/__images_here__'.
        - train_ratio: Ratio of the training set.
        - valid_ratio: Ratio of the validation set.
    Output:
        New directories for train, validation, and test sets.
        Eg: './data/processed/{time_create_dataset}_{train_ratio}_{valid_ratio}_{test_ratio}_{seed}/train',
            './data/processed/{time_create_dataset}_{train_ratio}_{valid_ratio}_{test_ratio}_{seed}/valid',
            './data/processed/{time_create_dataset}_{train_ratio}_{valid_ratio}_{test_ratio}_{seed}/test'.
        with
        './data/processed/{time_create_dataset}_{train_ratio}_{valid_ratio}_{test_ratio}_{seed}/train/B2 or B5 or B6' ...
    """
    # Check the data directory
    if not os.path.exists(data_dir):
        raise ValueError(f"Path {data_dir} does not exist.")

    # Check the ratios
    test_ratio = 1 - train_ratio - valid_ratio
    assert 0 <= train_ratio <= 1, "train_ratio must be in [0, 1]"
    assert 0 <= valid_ratio <= 1, "valid_ratio must be in [0, 1]"
    assert 0 <= test_ratio <= 1, "test_ratio must be in [0, 1]"

    # Create the destination directories
    begin_time = time.time()
    destination_dir = os.path.join(
        destination_dir,
        f"{int(begin_time)}_{int(train_ratio*100)}_{int(valid_ratio*100)}_{int(test_ratio*100)}_{seed}",
    )
    os.makedirs(destination_dir, exist_ok=True)
    os.makedirs(os.path.join(destination_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(destination_dir, "valid"), exist_ok=True)
    os.makedirs(os.path.join(destination_dir, "test"), exist_ok=True)

    # Get the list of images from subdirectories
    images = []
    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path):
            images += [
                os.path.join(subdir, f)
                for f in os.listdir(subdir_path)
                if os.path.isfile(os.path.join(subdir_path, f))
            ]

    if not images:
        raise ValueError(f"No files found in {data_dir}.")

    num_images = len(images)
    num_train = int(num_images * train_ratio)
    num_valid = int(num_images * valid_ratio)
    num_test = num_images - num_train - num_valid

    # Shuffle the images
    random.seed(seed)
    random.shuffle(images)

    # Split the images
    train_images = images[:num_train]
    valid_images = images[num_train : num_train + num_valid]
    test_images = images[num_train + num_valid :]

    # Copy the images to the destination directories
    for image in train_images:
        subdir, filename = os.path.split(image)
        os.makedirs(os.path.join(destination_dir, "train", subdir), exist_ok=True)
        shutil.copy(
            os.path.join(data_dir, image), os.path.join(destination_dir, "train", image)
        )
    for image in valid_images:
        subdir, filename = os.path.split(image)
        os.makedirs(os.path.join(destination_dir, "valid", subdir), exist_ok=True)
        shutil.copy(
            os.path.join(data_dir, image), os.path.join(destination_dir, "valid", image)
        )
    for image in test_images:
        subdir, filename = os.path.split(image)
        os.makedirs(os.path.join(destination_dir, "test", subdir), exist_ok=True)
        shutil.copy(
            os.path.join(data_dir, image), os.path.join(destination_dir, "test", image)
        )

    return (
        os.path.join(destination_dir, "train"),
        os.path.join(destination_dir, "valid"),
        os.path.join(destination_dir, "test"),
    )


####################################################################################################
# Augmentation image before train module 1                                                         #
####################################################################################################

import os
import random
import time
import cv2
from ultralytics import YOLO
from PIL import Image


# Augmentation Functions
def __save_plot_image(result, filename):
    result.save(filename=filename, labels=False, conf=False)


def __get_top_boxes_and_save_crop(result, image_A_set, output_dir, num_boxes=8):
    if isinstance(image_A_set, str):
        image_A_set = cv2.imread(image_A_set)
    if num_boxes > 8:
        raise ValueError("Number of boxes should be less than or equal to 8")
    boxes = result.boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        crops = []
        for i in range(2):
            for j in range(3):
                x1, y1 = j * 256, i * 256
                crops.append(image_A_set[y1 : y1 + 512, x1 : x1 + 512])
        crops.append(image_A_set[:768, :768])
        crops.append(image_A_set[:768, 256:1024])
    elif len(boxes) < num_boxes:
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        top_indices = areas.argsort()[::-1]
        crops = []
        for idx in top_indices:
            x1, y1, x2, y2 = boxes[idx].astype(int)
            crops.append(image_A_set[y1:y2, x1:x2])
        additional_needed = num_boxes - len(boxes)
        patch_options = [
            (0, 0, 512, 512),
            (256, 0, 768, 512),
            (512, 0, 1024, 512),
            (0, 256, 512, 768),
            (256, 256, 768, 768),
            (512, 256, 1024, 768),
            (0, 0, 768, 768),
            (256, 256, 1024, 1024),
        ]
        random_patches = random.sample(patch_options, additional_needed)
        for x1, y1, x2, y2 in random_patches:
            crops.append(image_A_set[y1:y2, x1:x2])
    else:
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        top_indices = areas.argsort()[-num_boxes:][::-1]
        crops = []
        for idx in top_indices:
            x1, y1, x2, y2 = boxes[idx].astype(int)
            crops.append(image_A_set[y1:y2, x1:x2])

    for i in range(8):
        crops[i] = cv2.resize(crops[i], (224, 224))
        cv2.imwrite(output_dir + f"_crop_{i}.jpg", crops[i])


def __get_patches_and_save(
    image_B_set_path,
    output_dir,
    num_patches=12,
    patch_size=(256, 256),
    resize_to=(224, 224),
):
    img = Image.open(image_B_set_path)
    img = img.resize((1024, 768))

    for i in range(3):
        for j in range(4):
            left = j * patch_size[0]
            upper = i * patch_size[1]
            right = left + patch_size[0]
            lower = upper + patch_size[1]

            patch = img.crop((left, upper, right, lower))
            patch = patch.resize(resize_to)
            patch.save(output_dir + f"_patch_{i*4+j+1}.jpg")


# Main Augmentation Logic
def augment_images(data_dir, destination_dir, model_path, batch_size=20):
    """
    Use: Augment the images in the data directory (only for train or valid set) x22 times
        - A set: original images (x1)
        - B set: original images with bounding boxes drawn (x1)
        - C set: original images cropped by bounding boxes (x8)
        - D set: images of B set with grid cropping (x12)
    Input:
        - data_dir: Path to the images directory.
                    Eg: 'data/B2/__images_here__',
                        'data/B5/__images_here__',
                        'data/B6/__images_here__'.
        - destination_dir: Path to the destination directory.
    Output:
        New directories for augmented images.
        Eg:
        data_dir = './data/processed/{dataset_name}/{dataset_subset}'
        destination_dir = './data/augmented'
        New augmented directories:
        './data/augmented/augmented_{dataset_subset}_{dataset_name}/B2 or B5 or B6' ...
    """
    if not os.path.exists(data_dir):
        raise ValueError(f"Path {data_dir} does not exist.")

    begin_time = time.time()
    dataset_subset = os.path.basename(data_dir)
    if dataset_subset not in ["train", "valid"]:
        raise ValueError(f"Only augmented train or valid set, not accepted test set.")
    dataset_name = data_dir.split("/")[-2]
    destination_dir = os.path.join(
        destination_dir, f"augmented_{dataset_subset}_{dataset_name}"
    )
    os.makedirs(destination_dir, exist_ok=True)

    # Get the list of images from subdirectories
    images = []
    for subdir in os.listdir(data_dir):
        os.makedirs(os.path.join(destination_dir, subdir), exist_ok=True)
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path):
            images += [
                os.path.join(subdir, f)
                for f in os.listdir(subdir_path)
                if os.path.isfile(os.path.join(subdir_path, f))
            ]
            # ['subdir/image1.jpg', 'subdir/image2.jpg', ...]

    if not images:
        raise ValueError(f"No files found in {data_dir}.")
    num_images = len(images)

    model = YOLO(model=model_path, task="detect")

    for batch_start in range(0, num_images, batch_size):
        batch_end = min(batch_start + batch_size, num_images)
        batch_images = images[batch_start:batch_end]
        batch_image_paths = [os.path.join(data_dir, img) for img in batch_images]

        results = model.predict(source=batch_image_paths, verbose=False)

        for i, result in enumerate(results):
            image_path = batch_image_paths[i]
            image_name = os.path.basename(image_path).replace(".jpg", "")
            subdir = os.path.dirname(image_path).split("/")[-1]

            destination_dir_subdir_name = os.path.join(
                destination_dir, subdir, image_name
            )

            # save A set x1
            shutil.copyfile(image_path, f"{destination_dir_subdir_name}_A.jpg")

            # save B set x1
            output_B_path = f"{destination_dir_subdir_name}_B.jpg"
            __save_plot_image(result=result, filename=output_B_path)

            # save C set x8
            output_C_path = f"{destination_dir_subdir_name}_C"
            __get_top_boxes_and_save_crop(
                result=result, image_A_set=image_path, output_dir=output_C_path
            )

            # save D set x12
            output_D_path = f"{destination_dir_subdir_name}_D"
            __get_patches_and_save(
                image_B_set_path=output_B_path, output_dir=output_D_path
            )

    print(f"Augmented the images in {data_dir}.")
    print(f"Time: {time.time() - begin_time} seconds.")
    return destination_dir


####################################################################################################
# Prepare data to train module 2                                                                   #
####################################################################################################

import os
import torch
import time
import csv
import numpy as np
from PIL import Image
from src.models.module1.cnn import get_cnn_model
from src.models.module1.vit import get_vit_model
from torchvision import transforms
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score


class PrepareDataTrainModule2:
    def __init__(
        self,
        data_dir,
        model_path,
        model_type="efficientnet_b0",
        device="cpu",
        num_classes=3,
    ):
        """
        This class is used to prepare the data for training the model in module 2.
        **Para one**: The path to folder dataset (train or val or test)
        It will take the folder directory of a set of data:
        Eg: 'data/processed/{dataset_name}/{dataset_subset}' like this
            'data/processed/1631550860_70_15_15_42/train' or
            'data/processed/1631550860_70_15_15_42/valid' or
            'data/processed/1631550860_70_15_15_42/test'.

        In each of these folder has the subfolders B2, B5, B6, etc. which contain the images.
        B2, B5, B6 are the classes of the images.

        **Para two**: The path to the model that will be used to predict label from one images.
        The second parameter is the path to a model that will be used to predict label from one images.
        Eg: 'model/best_f1_efficientNet_model.pth'

        **Output**:
        The output of this class is a dataframe that contains
            + the path to the images,
            + and the label of the images,
            + and the feature vector,
            + and the predicted vector.
        Eg:
        | path_to_image | label | feature_vector | predicted_vector |

        **How this works**:
        Step 1. Load the model
        Step 2. Load the data_dir
        Step 3. For each image in the data_dir, read the image.
        Step 4. Create a list []. Append the read image to the list.
        Step 5. From the image, crop 12 images as grid 4 columns and 3 rows.
        Step 6. Pass 13 images from a list to model.
        Step 7. Before pass to last dense layer -> get feature vector -> 13 vectors -> save this.
        Step 8. Pass feature vector to last dense layer -> 13 vectors -> save this.
        Step 9. Compare argmax(predicted label) to true label and then -> F1 score, accuracy, AUC, ...
        """
        self.set_data_dir(data_dir)
        self.model_map = {
            "vgg16": "vgg16",
            "vgg19": "vgg19",
            "resnet18": "resnet18",
            "resnet152": "resnet152",
            "densenet121": "densenet121",
            "densenet201": "densenet201",
            "efficientnet_b0": "efficientnet_b0",
            "efficientnet_b7": "efficientnet_b7",
            "mobilenet_v1": "mobilenet_v1",
            "mobilenet_v3_large": "mobilenet_v3_large",
            "vit_b_16": "vit_b_16",
            "vit_l_16": "vit_l_16",
        }
        self.set_model(
            model_path=model_path,
            model_type=model_type,
            device=device,
            num_classes=num_classes,
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    def set_data_dir(self, data_dir):
        print(f"Setting data directory to {data_dir}...")
        self.data_dir = data_dir

    def set_model(
        self, model_path, model_type="efficientnet_b0", device="cpu", num_classes=3
    ):
        print(
            f"Loading model {model_type} from {model_path} with {num_classes} classes..."
        )
        self.model_path = model_path
        self.model_type = model_type
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(
            model_path=model_path,
            model_type=model_type,
            device=self.device,
            num_classes=num_classes,
        )

    def __get_model_structure(self, model_type, num_classes):
        if model_type in self.model_map:
            if "vit" in model_type:
                return get_vit_model(
                    name=self.model_map[model_type], num_classes=num_classes
                )
            else:
                return get_cnn_model(
                    name=self.model_map[model_type], num_classes=num_classes
                )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def load_model(self, model_path, model_type, device, num_classes):
        print(f"Setting model structure...")
        model = self.__get_model_structure(
            model_type=model_type, num_classes=num_classes
        )
        print(f"Loading model weights, move to {device} and set eval mode...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model

    # def process(self, description: str = "Creator", path_save: str = "data.csv"):
    #     start_time = time.time()
    #     log = f"\nStart processing data for {description} when {start_time}..."
    #     print(log)
    #     logs = log
    #     all_data = []
    #     label_map = {"B2": 0, "B5": 1, "B6": 2}
    #     true_labels = []
    #     predicted_labels = []

    #     for label in os.listdir(self.data_dir):
    #         for image_name in os.listdir(os.path.join(self.data_dir, label)):
    #             image_path = os.path.join(self.data_dir, label, image_name)
    #             id_label = label_map[label]
    #             record = [image_path, id_label]
    #             log = f"\n{'#' * 20}\nProcessing {image_path} with label {id_label}..."
    #             print(log)
    #             logs += log

    #             # Read the origin image
    #             log = f"\nTaking 13 images from 1 image {image_path}..."
    #             print(log)
    #             logs += log
    #             origin_image = Image.open(image_path).convert("RGB")
    #             list_of_images = [origin_image]

    #             # Crop 12 images as grid from origin image
    #             list_of_images += self.__crop_12_patches(origin_image=origin_image)

    #             # Process images
    #             processed_images = torch.stack(
    #                 [self.transform(img) for img in list_of_images]
    #             ).to(self.device)

    #             # Pass 13 images to model to get feature vector
    #             log = f"\nPassing 13 images to model to get feature vector..."
    #             print(log)
    #             logs += log
    #             with torch.no_grad():
    #                 feature_vector = self.model.features(processed_images).cpu().numpy()
    #                 log = f"\nFeature vector shape: {feature_vector.shape}"
    #                 # log += f"\nFeature vector: {feature_vector}"
    #                 print(log)
    #                 logs += log
    #             record.append(feature_vector.tolist())

    #             # Pass feature vector to last dense layer to get predicted vector
    #             log = f"\nPassing feature vector to last dense layer to get predicted vector..."
    #             print(log)
    #             with torch.no_grad():
    #                 predicted_vector = (
    #                     self.model(processed_images).cpu().numpy()
    #                 )
    #                 log = f"\nPredicted vector shape: {predicted_vector.shape}"
    #                 log += f"\nPredicted vector: {predicted_vector}"
    #                 print(log)
    #                 logs += log
    #             record.append(predicted_vector.tolist())

    #             # Save the predicted label for the origin image
    #             predicted_label = np.argmax(predicted_vector[0])
    #             log = f"\nPredicted label: {predicted_label}"
    #             print(log)
    #             logs += log
    #             record.append(predicted_label.item())

    #             all_data.append(record)
    #             true_labels.append(id_label)
    #             predicted_labels.append(predicted_label)

    #     # Save the file as a csv
    #     print(f"Saving the data to {path_save}...")
    #     with open(path_save, "w") as f:
    #         f.write(
    #             "path_to_image,label,feature_vector,predicted_vector,predicted_label\n"
    #         )
    #         for record in all_data:
    #             f.write(
    #                 f"{record[0]},{record[1]},{record[2]},{record[3]},{record[4]}\n"
    #             )

    #     # Calculate the F1 score, accuracy, AUC
    #     f1 = f1_score(true_labels, predicted_labels, average="weighted")
    #     accuracy = accuracy_score(true_labels, predicted_labels)

    #     # Print the results
    #     log = f"\n{'#' * 20}\nData saved to {path_save}. Analyzing the results..."
    #     log += f"\nF1 score: {f1:.4f}"
    #     log += f"\nAccuracy: {accuracy:.4f}"
    #     log += f"\nAll done! in {time.time() - start_time:.2f} seconds.\n"
    #     print(log)
    #     logs += log

    #     # Save the logs
    #     with open("logs.txt", "a") as f:
    #         f.write(logs)
    #     return all_data

    def process(self, description: str = "Creator", path_save: str = "data.csv"):
        start_time = time.time()
        log = f"\nStart processing data for {description} when {start_time}..."
        print(log)
        logs = log

        # Mở tệp CSV và ghi header trước khi bắt đầu xử lý dữ liệu
        with open(path_save, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(
                [
                    "path_to_image",
                    "label",
                    "feature_vector",
                    "predicted_vector",
                    "predicted_label",
                ]
            )

            label_map = {"B2": 0, "B5": 1, "B6": 2}
            true_labels = []
            predicted_labels = []

            for label in os.listdir(self.data_dir):
                for image_name in os.listdir(os.path.join(self.data_dir, label)):
                    image_path = os.path.join(self.data_dir, label, image_name)
                    id_label = label_map[label]

                    origin_image = Image.open(image_path).convert("RGB")
                    list_of_images = [origin_image] + self.__crop_12_patches(
                        origin_image=origin_image
                    )
                    processed_images = torch.stack(
                        [self.transform(img) for img in list_of_images]
                    ).to(self.device)

                    # Get feature and predicted vectors
                    with torch.no_grad():
                        feature_vector = (
                            self.model.features(processed_images).cpu().numpy()
                        )
                        predicted_vector = self.model(processed_images).cpu().numpy()

                    predicted_label = np.argmax(predicted_vector[0])

                    # Ghi log ra tệp CSV từng hàng
                    csvwriter.writerow(
                        [
                            image_path,
                            id_label,
                            feature_vector.tolist(),
                            predicted_vector.tolist(),
                            predicted_label,
                        ]
                    )

                    true_labels.append(id_label)
                    predicted_labels.append(predicted_label)

                    log = f"\nProcessed {image_path} with id_label: {id_label}, predicted_label: {predicted_label} ..."
                    print(log)
                    logs += log

            # Save final metrics
            f1 = f1_score(true_labels, predicted_labels, average="weighted")
            accuracy = accuracy_score(true_labels, predicted_labels)

        # Save the logs
        with open(os.path.join(os.path.dirname(path_save), "logs.txt"), "a") as f:
            log = f"\nF1 score: {f1:.4f}\nAccuracy: {accuracy:.4f}\nAll done! in {time.time() - start_time:.2f} seconds.\n"
            f.write(logs + log)
        print(log)

        return all_data

    def __crop_12_patches(self, origin_image) -> list:
        width, height = origin_image.size
        patch_width = width // 4
        patch_height = height // 3
        patches = []

        for i in range(3):
            for j in range(4):
                left = j * patch_width
                top = i * patch_height
                right = left + patch_width
                bottom = top + patch_height
                patch = origin_image.crop((left, top, right, bottom))
                patches.append(patch)

        return patches

    @staticmethod
    def help_to_use():
        print("Here is an example to use:")
        print("data_dir = 'data/processed/1631550860_70_15_15_42/train'")
        print("model_path = 'model/best_f1_efficientNet_model.pth'")
        print("model_type = 'efficientnet_b0'")
        print("device = 'cuda:0'")
        print("num_classes = 3")
        print(
            "prepare_data = PrepareDataTrainModule2(data_dir, model_path, model_type, device, num_classes)"
        )
        print("prepare_data.process(path_save='data.csv')")
        print("If you want to know more about this class: ")
        print("PrepareDataTrainModule2.help_to_use()")
