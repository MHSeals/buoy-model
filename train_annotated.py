from ultralytics import YOLO
from roboflow import Roboflow

from secret import roboflow_api_key

rf = Roboflow(api_key=roboflow_api_key)

project = rf.workspace("mhseals").project("buoys-4naae")

dataset = project.version(6).download('yolov8')

model = YOLO("yolov8s.pt")

results = model.train(data=f"{dataset.location}/data.yaml",
                      imgsz=640,
                      epochs=600,
                      batch=8,
                      name=f"v{dataset.version}",
                      amp=True
                      )

model.export(format="torchscript")