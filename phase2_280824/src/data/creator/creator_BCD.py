# src/data/creator/creator_BCD.py
import glob
import os
import cv2
from ultralytics import YOLO
from PIL import Image
import random
import time


# 1. Get and save the plot images (B)
def save_plot_image(result, output_dir):
    img = result.plot(labels=False, conf=False)
    cv2.imwrite(output_dir, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# 2. Get top 8 bounding boxes with max area, crop from A set and save them (C)
def get_top_boxes_and_save_crop(result, image_A_set, output_dir, num_boxes=8):
    if isinstance(image_A_set, str):
        image_A_set = cv2.imread(image_A_set)
    if num_boxes > 8:
        raise ValueError("Number of boxes should be less than or equal to 8")
    boxes = result.boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        # Case: No bounding boxes detected
        crops = []
        # 6 patches of size 512x512
        for i in range(2):
            for j in range(3):
                x1, y1 = j * 256, i * 256
                crops.append(image_A_set[y1 : y1 + 512, x1 : x1 + 512])
        # 2 patches of size 768x768
        crops.append(image_A_set[:768, :768])
        crops.append(image_A_set[:768, 256:1024])
    elif len(boxes) < num_boxes:
        # Case: Less than 8 bounding boxes detected
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        top_indices = areas.argsort()[::-1]
        crops = []
        for idx in top_indices:
            x1, y1, x2, y2 = boxes[idx].astype(int)
            crops.append(image_A_set[y1:y2, x1:x2])

        # Add random patches to make up the difference
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
        # Case: 8 or more bounding boxes detected
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        top_indices = areas.argsort()[-num_boxes:][::-1]
        crops = []
        for idx in top_indices:
            x1, y1, x2, y2 = boxes[idx].astype(int)
            crops.append(image_A_set[y1:y2, x1:x2])

    for i in range(8):
        crops[i] = cv2.resize(crops[i], (224, 224))
        # Save the crop
        cv2.imwrite(output_dir + f"_crop_{i}.jpg", crops[i])


# 3. Get 12 patches from image B, resize, and save (D)
def get_patches_and_save(
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


def __create_folder(config, dataset_dir):
    for label in config["data"]["class"]:
        for split in ["train", "valid", "test"]:
            folder_path = os.path.join(dataset_dir, split, label)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)


def run(config, A_set_dir:str):
    if not config["creator"]["A_set"]:
        print("Skipping data B and C creator as A set is not created")
        return
    start_time = time.time()
    print("Running data BCD creator")
    model = YOLO(
        model=config["model"]["path"],
        task=config["model"]["task"],
        verbose=config["model"]["verbose"],
    )
    print("Loaded cluster detect model")

    data_input = []
    train_path = os.path.join(A_set_dir, "train")
    valid_path = os.path.join(A_set_dir, "valid")
    for label in config["data"]["class"]:
        data_input += glob.glob(os.path.join(train_path, label, "*.jpg"))
        data_input += glob.glob(os.path.join(valid_path, label, "*.jpg"))
    print("Loaded data dir input")

    batch_size = config["model"]["batch_size"]
    num_images = len(data_input)
    print(f"Total images: {num_images}")
    print(f"Batch size: {batch_size}")

    B_set_dir = A_set_dir.replace("A", "B")
    C_set_dir = A_set_dir.replace("A", "C")
    D_set_dir = A_set_dir.replace("A", "D")
    if config["creator"]["B_set"]:
        __create_folder(config=config, dataset_dir=B_set_dir)
    if config["creator"]["C_set"]:
        __create_folder(config=config, dataset_dir=C_set_dir)
    if config["creator"]["D_set"]:
        __create_folder(config=config, dataset_dir=D_set_dir)
    print("Created output folders")

    for num_step in range(0, num_images // batch_size + 1):
        start_index = num_step * batch_size
        end_index = min(num_images, (num_step + 1) * batch_size)
        image_paths = data_input[start_index:end_index]

        # print(image_paths)
        results = model.predict(source=image_paths, verbose=config["model"]["verbose"])
        print(f"Predicted batch {num_step+1}/{num_images//batch_size+1}")

        for i, result in enumerate(results):
            image_path = image_paths[i]
            destination = "/".join(
                image_path.split("/")[-3:],
            )   # part/label/img_name.jpg
            if config["creator"]["B_set"]:
                output_dir = os.path.join(B_set_dir, destination)
                save_plot_image(result=result, output_dir=output_dir)
            if config["creator"]["C_set"]:
                output_dir = os.path.join(C_set_dir, destination)
                get_top_boxes_and_save_crop(
                    result=result,
                    image_A_set=image_path,
                    output_dir=output_dir,
                    num_boxes=8,
                )
            if config["creator"]["D_set"]:
                output_dir = os.path.join(D_set_dir, destination)
                get_patches_and_save(
                    image_B_set_path=image_B_set_path, output_dir=output_dir
                )
            print(f"Processed image {i+1}/{len(results)}")

    print("Data creation complete in", time.time() - start_time, "seconds")
