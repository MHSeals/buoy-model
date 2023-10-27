from ultralytics import YOLO
import cv2
import os

model = YOLO("runs/detect/v9/weights/best.pt")

dir = "Buoys-9/test/images"

image_list = os.listdir(dir)

images = [cv2.imread(os.path.join(dir, file)) for file in image_list]

results = {}

# Use while loop with count variable to allow backtracking

it = 0

while True:
    frame = images[it]
    
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    if results.get(image_list[it]) is None:
        result = model(frame, save_conf=True)
        results[image_list[it]] = result
    else:
        result = results[image_list[it]]

    for pred in result:
        names = pred.names
        print(names)
        
        for i in range(len(pred.boxes)):
            name = names.get(int(pred.boxes.cls[i]))
            confidence = pred.boxes.conf[i]
            bounding_box = pred.boxes[i].xyxy[0]

            print(f"{name} {int(confidence*100)}% {bounding_box}")

    frame = result[0].plot(font_size=0.5, line_width=1, pil=False)

    cv2.rectangle(frame, (0, 0), (130, 30), (255, 255, 255), -1)
    cv2.putText(frame, f"Image {it+1}/{len(images)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow("result", frame)
    c = cv2.waitKey(1)
    # TAB, BACKTICK (`)
    while not c == 9 and not c == 96:
        c = cv2.waitKey(1)
        if c == 27:
            exit(-1)
        if cv2.getWindowProperty("result", cv2.WND_PROP_VISIBLE) < 1:
            exit(-1)
    
    if c == 9 and not it >= len(images)-1:
        it += 1
    elif c == 9 and it >= len(images)-1:
        it = 0

    if c == 96 and not it <= 0:
        print(3)
        it -= 1
    elif c == 96 and it <= 0:
        it = len(images) - 1

    print(it)
