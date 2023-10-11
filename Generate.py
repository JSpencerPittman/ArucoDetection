import cv2
import os
import yaml
from Tag import Tag
import numpy as np

with open("GenerateConfig.yml", "r") as file:
    config = yaml.safe_load(file)

def generate_tag(tag_idx: int, image):
    new_tag = Tag(config["ID"][tag_idx], 
                  config["CENTER_X"][tag_idx],
                  config["CENTER_Y"][tag_idx],
                  config["SIZE"][tag_idx],
                  config["OPACITY"][tag_idx],
                  config["BLUR"][tag_idx])

    new_tag.draw_marker(image)

def generate():
    image_dims = (config["IMAGE_WIDTH"], config["IMAGE_HEIGHT"], 3)
    if config["BACKGROUND_IMAGE"]:
        image = cv2.imread(config["BACKGROUND_IMAGE"], cv2.IMREAD_COLOR)
        image = cv2.resize(image, image_dims)
    else:
        image = 255 * np.ones(image_dims, dtype=np.uint8)

    tag_cnt = len(config["ID"])

    for tag_idx in range(tag_cnt):
        generate_tag(tag_idx, image)
    
    cv2.imwrite(config["OUTPUT_PATH"], image)


if __name__ == "__main__":
    generate()
