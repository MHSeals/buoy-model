<a href="https://universe.roboflow.com/mhseals/buoys-4naae">
    <img src="https://app.roboflow.com/images/download-dataset-badge.svg"></img>
</a><a href="https://universe.roboflow.com/mhseals/buoys-4naae/model/">
    <img src="https://app.roboflow.com/images/try-model-badge.svg"></img>
</a>

# This is a repository to train models on top of YoloV8. It also contains tools to run the model.
(sort of) depends on roboflow. You could remove the roboflow-specific code and use your own dataset.

## How to use
1. Clone the repository
- `git clone https://github.com/MHSeals/buoy-model.git`
2. Install the requirements
- `pip install -r requirements.txt`
3. Train the model
- `python train_annotated.py`
  - This will attempt to download the dataset from roboflow. If you want to use your own dataset, you can modify the `train_annotated.py` file to use your own dataset by passing an instance of the `Dataset` class and removing the roboflow specific code.
4. Retrieve the model weights
- Ultralytics is kind of wierd in that it doesn't always save the weights to the same place. It will be in a folder called `runs/detect/<version name>/weights/best.pt`. This will probably either be in the root of this project, or in the root of your python installation.
5. Run the model
- This step will change soon as I work on a better way to run the model. For now, you can run the model by following the instructions below.
- There are 3 available run modes. You can run on test mode (a folder of images and you click between the images), you can run detection, or you can run tracking.
  - test mode: `python detect_test.py`
  - detection mode: `python detect_webcam.py`
  - tracking mode: `python detect_tracking.py`