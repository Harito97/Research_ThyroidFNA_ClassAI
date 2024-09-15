# process.py content
# Used to process the data for the Thyroid FNA classification project.

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
