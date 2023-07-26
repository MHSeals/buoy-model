from ultralytics import YOLO
import os
# from roboflow import Roboflow

# from secret import roboflow_api_key

# rf = Roboflow(api_key=roboflow_api_key)

# project = rf.workspace("mhseals").project("buoys-4naae")

# dataset = project.version(7).download("yolov8")

class Dataset:
    def __init__(self, location, version):
        self.location = os.path.abspath(location)
        self.version = version


dataset = Dataset(location="SYNTHETIC-YOLO", version=1)

model = YOLO("yolov8s.pt")

results = model.train(
    data=f"{dataset.location}/data.yaml",
    imgsz=640,
    epochs=600,
    batch=8,
    name=f"v{dataset.version}",
    amp=True,
)
