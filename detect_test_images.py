from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
import os

from rgb import rgb, colors

model = YOLO("./v13.pt")

dir = "./Buoys-13/test/images"

image_list = os.listdir(dir)

images = [cv2.imread(os.path.join(dir, file)) for file in image_list]

results = {}

FRAME_SIZE = (1280, 720)

IN_SIZE = (1280, 1280)

# Use while loop with count variable to allow backtracking

it = 0

while True:
    frame = images[it]

    frame = cv2.resize(frame, FRAME_SIZE)

    original_frame = frame.copy()

    x_scale_factor = frame.shape[1] / IN_SIZE[0]
    y_scale_factor = frame.shape[0] / IN_SIZE[1]
    x_orig, y_orig = frame.shape[1], frame.shape[0]

    if results.get(image_list[it]) is None:
        frame = cv2.resize(frame, IN_SIZE)
        result = model.predict(frame)
        results[image_list[it]] = result
    else:
        result = results[image_list[it]]

    frame_area = frame.shape[0] * frame.shape[1]

    for pred in results[image_list[it]]:
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

            area = int(bounding_box[2] - bounding_box[0]) * (bounding_box[3] - bounding_box[1])

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

    cv2.rectangle(original_frame, (0, 0), (130, 30), (255, 255, 255), -1)
    cv2.putText(original_frame, f"Image {it+1}/{len(images)}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow("result", original_frame)
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
        it -= 1
    elif c == 96 and it <= 0:
        it = len(images) - 1

    print(it)
