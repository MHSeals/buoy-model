from roboflow import Roboflow

from secret import roboflow_api_key

rf = Roboflow(api_key=roboflow_api_key)

project = rf.workspace("mhseals").project("buoys-4naae")

project.version(13).deploy(model_type="yolov8", model_path=f"/home/alec/.pyenv/runs/detect/v13/")
