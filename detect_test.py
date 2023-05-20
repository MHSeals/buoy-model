from ultralytics import YOLO
import cv2
import os

model = YOLO("runs/detect/v5_300e/weights/best.pt")

images = [cv2.imread(f"test_images/{file}") for file in os.listdir("test_images")]

for i in range(len(images)):
    frame = images[i]
    
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    result = model(frame, save_conf=True)

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
    
    cv2.imshow("result", frame)
    c = cv2.waitKey(1)
    while not c == 9:
        c = cv2.waitKey(1)
        if cv2.getWindowProperty("result", cv2.WND_PROP_VISIBLE) < 1:
            exit(-1)
