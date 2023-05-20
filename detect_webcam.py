from ultralytics import YOLO
import cv2
import time

model = YOLO("runs/detect/v5_300e/weights/best.pt")

cap = cv2.VideoCapture("/dev/video0") # can change to use different webcams

if not cap.isOpened():
    raise IOError("Cannot open webcam")

start_time = time.perf_counter()

display_time = 1
fc = 0
FPS = 0

while True:
    _, frame = cap.read()
    fc += 1
    TIME = time.perf_counter() - start_time

    if (TIME) >= display_time :
        FPS = fc / (TIME)
        fc = 0
        start_time = time.perf_counter()

    fps_disp = "FPS: "+str(FPS)[:5]
    
    #frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    result = model(frame)

    for pred in result:
        names = pred.names
        print(names)

        for i in range(len(pred.boxes)):
            name = names.get(int(pred.boxes.cls[i]))
            confidence = pred.boxes.conf[i]
            bounding_box = pred.boxes[i].xyxy[0]

            print(f"{name} {int(confidence)}% {bounding_box}")

            frame = cv2.putText(frame, f"{name} {int(confidence*100)}%", (int(bounding_box[0]), int(bounding_box[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            frame = cv2.rectangle(frame, (int(bounding_box[0]), int(bounding_box[1])), (int(bounding_box[2]), int(bounding_box[3])), (0, 255, 0), 1)

    frame = cv2.putText(frame, fps_disp, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("result", frame)
    c = cv2.waitKey(1)
    if c == 27:
        break

    if cv2.getWindowProperty("result", cv2.WND_PROP_VISIBLE) < 1:
        break
