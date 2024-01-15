from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
import time
from collections import defaultdict

from rgb import rgb, colors

model = YOLO("./best.pt")

track_history = defaultdict(lambda: [])

cap = cv2.VideoCapture("PXL_20240108_222954915.TS.mp4")  # can change to use different webcams

if not cap.isOpened():
    raise IOError("Cannot open webcam")

start_time = time.perf_counter()

display_time = 1
fc = 0
FPS = 0
total_frames = 0
prog_start = time.perf_counter()

FRAME_SIZE = (1280, 720)

IN_SIZE = (1280, 1280)

frame = cap.read()[1]
frame = cv2.resize(frame, FRAME_SIZE)
x_scale_factor = frame.shape[1] / IN_SIZE[0]
y_scale_factor = frame.shape[0] / IN_SIZE[1]
x_orig, y_orig = frame.shape[1], frame.shape[0]


while True:
    total_frames += 1
    TIME = time.perf_counter() - start_time
    success, frame = cap.read()

    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    original_frame = frame.copy()
    original_frame = cv2.resize(original_frame, FRAME_SIZE)
    frame = cv2.resize(frame, IN_SIZE)

    frame_area = frame.shape[0] * frame.shape[1]

    fc += 1

    if (TIME) >= display_time:
        FPS = fc / (TIME)
        fc = 0
        start_time = time.perf_counter()

    fps_disp = "FPS: "+str(FPS)[:5]

    results = model.predict(frame)

    original_frame = cv2.putText(
        original_frame, fps_disp, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    original_frame = cv2.putText(original_frame, "Press k to pause", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    original_frame = cv2.putText(original_frame, "Press ESC to exit", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    original_frame = cv2.putText(original_frame, "Press r to restart", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    for pred in results:
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

            x, y = int(bounding_box[0]), int(bounding_box[1])
            w, h = int(bounding_box[2] - bounding_box[0]), int(bounding_box[3] - bounding_box[1])

            # Calculate area of bounding box

            area = (bounding_box[2] - bounding_box[0]) * (bounding_box[3] - bounding_box[1])

            # Disregard large bounding boxes

            if area / frame_area > 0.20:
                continue

            color = colors.get(name, rgb(255, 255, 255))

            print(f"{name} {int(confidence*100)}% {bounding_box}")

            # original_frame = cv2.putText(original_frame, 
            #                              f"{name} ({int(confidence*100)})% {int(area)}px",
            #                              (int(bounding_box[0]), int(bounding_box[1])-5),
            #                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color.as_bgr(), 1)
            # original_frame = cv2.rectangle(original_frame,
            #                                (int(bounding_box[0]), int(bounding_box[1])),
            #                                (int(bounding_box[2]), int(bounding_box[3])), 
            #                                color.as_bgr(), 1)

            annotator = Annotator(original_frame, line_width=1)

            annotator.box_label((x, y, x+w, y+h), f"{name} ({int(confidence*100)})% {int(area)}px",
                                color=color.as_bgr(), txt_color=color.text_color().as_bgr())

            original_frame = annotator.result()

    cv2.imshow("result", original_frame)
    c = cv2.waitKey(1)
    if c == 107:
        time.sleep(0.1)
        while True:
            c = cv2.waitKey(1)
            if c == 107 or c == 27:
                break

            if cv2.getWindowProperty("result", cv2.WND_PROP_VISIBLE) < 1:
                break

    if c == 27:
        break

    if cv2.getWindowProperty("result", cv2.WND_PROP_VISIBLE) < 1:
        break

    if c == 114:
        track_history.clear()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

cap.release()
cv2.destroyAllWindows()

print(f"Avg FPS: {total_frames / (time.perf_counter() - prog_start)}")

# AVG FPS: 28.414