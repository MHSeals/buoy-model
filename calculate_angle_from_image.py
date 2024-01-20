import cv2
import math
import numpy as np

# Realsense D435 FOV (degrees)
FOV_H = 69
FOV_V = 42

def calculateAngleOfPixel(frame: np.ndarray, pixel: tuple[int, int]) -> tuple[float, float]:
    height, width, _ = frame.shape
    x, y = pixel

    degreesPerPixelH = FOV_H / width
    degreesPerPixelV = FOV_V / height

    x_from_center = x - width/2
    y_from_center = y - height/2

    angleH = x_from_center * degreesPerPixelH
    angleV = y_from_center * degreesPerPixelV

    return angleH, angleV

cap = cv2.VideoCapture("/dev/video6") # can change to use different webcams

if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    _, frame = cap.read()
    height, width, _ = frame.shape
    print(height, width)

    angle = calculateAngleOfPixel(frame, (width//2, height//2))
    print(angle)

    cv2.imshow("Input", frame)
    c = cv2.waitKey(1)
    if c == 27:
        break
