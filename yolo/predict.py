import os
from PIL import Image
from ultralytics import YOLO

IMAGE_DIR = "realworld"
SAVE_DIR = "results"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

model = YOLO("./weights/best.pt")

save_paths = [os.path.join(SAVE_DIR, p) for p in os.listdir(IMAGE_DIR)]

results = model(IMAGE_DIR)
for path, res in zip(save_paths, results):
    print(path)
    im_array = res.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.save(path)