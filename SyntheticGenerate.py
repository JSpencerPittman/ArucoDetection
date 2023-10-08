import cv2
import os
import random
import yaml
from Tag import Tag  # Assuming the previously translated Tag class is saved as Tag.py

CLASS_ID = 0

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)


def generate_tag():
    tag_id = random.randint(config["MIN_ID"], config["MAX_ID"])
    opacity = random.randint(config["MIN_OPACITY"], config["MAX_OPACITY"])

    blur = 0
    if random.choice([True, False]):
        blur = random.randint(config["MIN_BLUR"], config["MAX_BLUR"])

    size = random.randint(config["MIN_SIZE"], config["MAX_SIZE"])
    low_bound = size // 2
    high_bound = config["IMAGE_SIZE"] - size // 2
    center_x = random.randint(low_bound, high_bound)
    center_y = random.randint(low_bound, high_bound)

    return Tag(tag_id, center_x, center_y, size, opacity, blur)


def create_tag_string(tag):
    cx_norm = tag.x() / config["IMAGE_SIZE"]
    cy_norm = tag.y() / config["IMAGE_SIZE"]
    size_norm = tag.size() / config["IMAGE_SIZE"]

    return f"{CLASS_ID} {cx_norm:.6f} {cy_norm:.6f} {size_norm:.6f} {size_norm:.6f}\n"


def generate(image_save_dir, annot_save_dir, image_count):
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)
    if not os.path.exists(annot_save_dir):
        os.makedirs(annot_save_dir)

    background_paths = [os.path.join(config["BACKGROUND_IMAGE_DIR"], f) for f in
                        os.listdir(config["BACKGROUND_IMAGE_DIR"])]

    for img_idx in range(image_count):
        filename = f"image_{img_idx}"

        annot_save_path = os.path.join(annot_save_dir, f"{filename}.txt")
        image_save_path = os.path.join(image_save_dir, f"{filename}.png")

        background_path = random.choice(background_paths)
        image = cv2.imread(background_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (config["IMAGE_SIZE"], config["IMAGE_SIZE"]))
        tag_count = random.randint(config["MIN_TAG_COUNT"], config["MAX_TAG_COUNT"])

        tags = []
        with open(annot_save_path, "w") as annot_file:
            for _ in range(tag_count):
                new_tag = generate_tag()
                overlap = any(Tag.is_overlap(tag, new_tag) for tag in tags)
                if not overlap:
                    annot_file.write(create_tag_string(new_tag))
                    tags.append(new_tag)
                    new_tag.draw_marker(image)

        cv2.imwrite(image_save_path, image)
        print(image_save_path, annot_save_path)


if __name__ == "__main__":
    # training dataset
    generate(
        os.path.join(config["DATA_PATH"], "images/train/"),
        os.path.join(config["DATA_PATH"], "labels/train/"),
        config["NUM_TRAIN_IMAGES"]
    )

    # test dataset
    generate(
        os.path.join(config["DATA_PATH"], "images/test/"),
        os.path.join(config["DATA_PATH"], "labels/test/"),
        config["NUM_TEST_IMAGES"]
    )
