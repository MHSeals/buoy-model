from ultralytics import YOLO
from roboflow import Roboflow
from datetime import datetime
import csv
import os
import yaml

from secret import roboflow_api_key
from filter_classes import filter_classes, remap_classes

# Remap classes from racquet balls to buoys for training.
# Since racquetballs are not currently required to be detected
# Mapping them as buoys helps with detecting buoys from farther distances
REMAP_CLASSES = {
    "blue_racquet_ball": "blue_buoy",
    "red_racquet_ball": "red_buoy",
    "yellow_racquet_ball": "yellow_buoy",
}

# Classes to keep during training. Set to None to keep all classes.
KEEP_CLASSES = [
    "black_buoy",
    "black_cross",
    "black_target_boat",
    "black_triangle",
    "blue_buoy",
    "green_buoy",
    "green_light_buoy",
    "green_pole_buoy",
    "red_buoy",
    "red_light_buoy",
    "red_pole_buoy",
    "yellow_buoy",
    "yellow_target_boat",
]

rf = Roboflow(api_key=roboflow_api_key)

project = rf.workspace("mhseals").project("buoys-4naae")

version = project.version(20)

print(
    f"Downloading version {version.version} of {project.name} created at {datetime.fromtimestamp(version.created)}"
)

dataset = version.download("yolov11", overwrite=False)

if REMAP_CLASSES:
    print(f"\nRemapping classes: {REMAP_CLASSES}")
    remap_classes(dataset.location, REMAP_CLASSES)

if KEEP_CLASSES is not None:
    print(f"\nFiltering dataset to classes: {KEEP_CLASSES}")
    filter_classes(dataset.location, KEEP_CLASSES)

with open(os.path.join(dataset.location, "data.yaml"), "r") as f:
    final_classes: list[str] = yaml.safe_load(f)["names"]

train_name = f"v{dataset.version}"
train_dir = os.path.join("runs", "detect", train_name)
os.makedirs(train_dir, exist_ok=True)

csv_path = os.path.join(train_dir, "classes.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["class_id", "class_name"])
    writer.writerows(enumerate(final_classes))

print(f"Saved class list to {csv_path}")

model = YOLO("runs/detect/v20/weights/last.pt")

results = model.train(
    data=f"{dataset.location}/data.yaml",
    imgsz=640,
    epochs=300,  # Long training run to capture full learning curve
    batch=20,  # Increased from 8 to better utilize GPU
    patience=0,  # Disable early stopping to train full duration
    save_period=10,  # Save checkpoint every 10 epochs
    name=train_name,
    amp=True,
    cache="ram",  # Use RAM caching for faster data loading
    workers=12,  # Increased workers for better data pipeline
    resume=True,
)
