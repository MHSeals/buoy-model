from ultralytics import YOLO
from roboflow import Roboflow
from datetime import datetime
import os

from secret import roboflow_api_key

rf = Roboflow(api_key=roboflow_api_key)

project = rf.workspace("mhseals").project("buoys-4naae")

version = project.version(20)

print(f"Downloading version {version.version} of {project.name} created at {datetime.fromtimestamp(version.created)}")

dataset = version.download("yolov11", overwrite=False)

class Dataset:
    def __init__(self, location, version):
        self.location = os.path.abspath(location)
        self.version = version


# dataset = Dataset(location="SYNTHETIC-YOLO", version=1)

model = YOLO("runs/detect/v20/weights/last.pt")

results = model.train(
    data=f"{dataset.location}/data.yaml",
    imgsz=640,
    epochs=300,          # Long training run to capture full learning curve
    batch=20,            # Increased from 8 to better utilize GPU
    patience=0,          # Disable early stopping to train full duration
    save_period=10,      # Save checkpoint every 10 epochs
    name=f"v{dataset.version}",
    amp=True,
    cache='ram',         # Use RAM caching for faster data loading
    workers=12,          # Increased workers for better data pipeline
    resume=True,
)
