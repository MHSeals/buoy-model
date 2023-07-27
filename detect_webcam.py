from ultralytics import YOLO
import cv2
import time
from enum import Enum

model = YOLO("runs/detect/v7/weights/best.pt")

cap = cv2.VideoCapture("/dev/video0") # can change to use different webcams

if not cap.isOpened():
    raise IOError("Cannot open webcam")

start_time = time.perf_counter()

display_time = 1
fc = 0
FPS = 0

# Hack to get color picker inside vscode
class rgb():
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b
    
    def __str__(self):
        return f"rgb({self.r}, {self.g}, {self.b})"

    def __repr__(self):
        return f"rgb({self.r}, {self.g}, {self.b})"
    
    def as_bgr(self) -> tuple:
        return (self.b, self.g, self.r)

colors: dict[rgb] = {
    "blue_ball": rgb(0, 0, 255),
    "dock": rgb(109, 67, 3),
    "green_ball": rgb(0, 255, 0),
    "green_pole": rgb(0, 255, 0),
    "misc_buoy": rgb(0, 217, 255),
    "red_ball": rgb(255, 0, 0),
    "red_pole": rgb(255, 0, 0),
    "yellow_ball": rgb(255, 255, 0),
}


while True:
    TIME = time.perf_counter() - start_time
    _, frame = cap.read()
    original_frame = frame.copy()
    x_scale_factor = original_frame.shape[1] / 640
    y_scale_factor = original_frame.shape[0] / 640
    x_orig, y_orig = original_frame.shape[1], original_frame.shape[0]
    frame = cv2.resize(frame, (640, 640))
    fc += 1

    if (TIME) >= display_time :
        FPS = fc / (TIME)
        fc = 0
        start_time = time.perf_counter()

    fps_disp = "FPS: "+str(FPS)[:5]
    
    #frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    result = model(frame)

    for pred in result:
        names = pred.names

        for i in range(len(pred.boxes)):
            name = names.get(int(pred.boxes.cls[i]))
            confidence = pred.boxes.conf[i]
            bounding_box = pred.boxes[i].xyxy[0]
            bounding_box = [
                bounding_box[0] * x_scale_factor,
                bounding_box[1] * y_scale_factor,
                bounding_box[2] * x_scale_factor,
                bounding_box[3] * y_scale_factor
            ]

            print(f"{name} {int(confidence*100)}% {bounding_box}")

            color = colors.get(name, rgb(255, 255, 255))

            original_frame = cv2.putText(original_frame, 
                                         f"{name} {int(confidence*100)}%",
                                         (int(bounding_box[0]), int(bounding_box[1])-5),
                                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, color.as_bgr(), 1)
            original_frame = cv2.rectangle(original_frame,
                                           (int(bounding_box[0]), int(bounding_box[1])),
                                           (int(bounding_box[2]), int(bounding_box[3])), 
                                           color.as_bgr(), 1)

    original_frame = cv2.putText(original_frame, fps_disp, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("result", original_frame)
    c = cv2.waitKey(1)
    if c == 27:
        break

    if cv2.getWindowProperty("result", cv2.WND_PROP_VISIBLE) < 1:
        break
