from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
import time
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from rgb import rgb, colors
from image_preprocessing import preprocess

import argparse

# Parse args early to check if we're in output mode
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", help="Path to video file")
parser.add_argument("-vf", "--video_fps", type=float, help="FPS limiter (default: uncapped)", default=None)
parser.add_argument("-o", "--output", help="Path to output video file", default=None)
parser.add_argument("-r", "--realsense", action="store_true", help="Use realsense camera")
parser.add_argument("-d", "--device", type=int, help="Device number of camera", default=0)
parser.add_argument("-s", "--display_size", type=int, nargs=2, help="Display size", default=(1280, 720))
parser.add_argument("-i", "--input_size", type=int, nargs=2, help="Input size", default=(1080, 1080))

args = parser.parse_args()

print("Loading model...")
model = YOLO("./best.pt")


track_history = defaultdict(lambda: [])

class RealsenseCapture:
    def __init__(self):
        import pyrealsense2 as rs

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

        self.pipeline.start(self.config)

        self._is_opened = True

    def read(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        frame = np.asanyarray(color_frame.get_data())
        return True, frame

    def release(self):
        self._is_opened = False
        self.pipeline.stop()

    def isOpened(self):
        return self._is_opened

# USE_REALSENSE = False

# DISP_SIZE = (1280, 720)

# IN_SIZE = (1080, 1080)

# TARGET_FPS = 2.8798397863818423

# if USE_REALSENSE:
#     cap = RealsenseCapture()

# else:
#     # can change to use different webcams
#     cap = cv2.VideoCapture("video.mp4")

DISP_SIZE = tuple(args.display_size)
IN_SIZE = tuple(args.input_size)
TARGET_FPS = args.video_fps

print("Getting video capture...")
if args.video is not None:
    cap = cv2.VideoCapture(args.video)
elif args.realsense:
    cap = RealsenseCapture()
else:
    cap = cv2.VideoCapture(args.device)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Get total frame count for progress bar when processing video files
total_video_frames = None
if args.video is not None and args.output is not None:
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

start_time = time.perf_counter()

display_time = 1
fc = 0
FPS = 0
total_frames = 0
prog_start = time.perf_counter()

frame = cap.read()[1]
frame = cv2.resize(frame, DISP_SIZE)
x_scale_factor = frame.shape[1] / IN_SIZE[0]
y_scale_factor = frame.shape[0] / IN_SIZE[1]
x_orig, y_orig = frame.shape[1], frame.shape[0]

# Initialize video writer if output file specified
video_writer = None
if args.output is not None:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_fps = args.video_fps if args.video_fps is not None else 30.0
    video_writer = cv2.VideoWriter(args.output, fourcc, output_fps, DISP_SIZE)

# create overlay for on screen text

overlay = np.zeros_like(frame)

overlay = cv2.putText(overlay, "Press k to pause",
                                 (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

overlay = cv2.putText(overlay, "Press ESC to exit",
                                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

overlay = cv2.putText(overlay, "Press r to restart (video cap only)", (
    10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Initialize progress bar for video file output
pbar = None
if args.output is not None and total_video_frames is not None:
    pbar = tqdm(total=total_video_frames, desc="Processing", unit="frames")

while True:
    frame_start_time = time.perf_counter()
    total_frames += 1
    TIME = time.perf_counter() - start_time
    success, frame = cap.read()

    if not success:
        # Don't loop if saving to file
        if args.output is not None:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame = cv2.resize(frame, DISP_SIZE)

    # frame = preprocess(frame)

    original_frame = frame.copy()
    
    frame = cv2.resize(frame, IN_SIZE)

    frame_area = frame.shape[0] * frame.shape[1]

    fc += 1

    if (TIME) >= display_time:
        FPS = fc / (TIME)
        fc = 0
        start_time = time.perf_counter()

    fps_disp = "FPS: "+str(FPS)[:5]

    frame = preprocess(frame)

    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=(args.output is None))

    original_frame = cv2.putText(
        original_frame, fps_disp, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    original_frame = cv2.addWeighted(original_frame, 1, overlay, 0.5, 0)

    for pred in results:
        names = pred.names

        # TODO: sometimes, on a frame with lots of objects, the model will only detect 1 object
        # for some reason, it is always on the same frames

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

            # Calculate area of bounding box

            area = (bounding_box[2] - bounding_box[0]) * \
                (bounding_box[3] - bounding_box[1])

            # Disregard large bounding boxes

            if area / frame_area > 0.20:
                continue

            x, y = bounding_box[:2]
            w, h = bounding_box[2] - x, bounding_box[3] - y

            center_x = x + w / 2
            center_y = y + h / 2

            id = None

            color = colors.get(name, rgb(255, 255, 255))

            if pred.boxes.id is not None:
                id = int(pred.boxes.id[i])

                track = track_history[id]
                track.append((float(center_x), float(center_y)))
                if len(track) > 30:
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                original_frame = cv2.polylines(
                    original_frame, [points], isClosed=False, color=color.as_bgr(), thickness=2)

            if args.output is None:
                print(f"{name} {int(confidence*100)}% {bounding_box}")

            # original_frame = cv2.putText(original_frame,
            #                              f"{id if id is not None else 'None'}: {name} ({int(confidence*100)})% {int(area)}px",
            #                              (int(bounding_box[0]), int(bounding_box[1])-5),
            #                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color.as_bgr(), 1)
            # original_frame = cv2.rectangle(original_frame,
            #                                (int(bounding_box[0]), int(bounding_box[1])),
            #                                (int(bounding_box[2]), int(bounding_box[3])),
            #                                color.as_bgr(), 1)

            annotator = Annotator(original_frame, line_width=1)

            annotator.box_label((x, y, x+w, y+h), f"{id if id is not None else 'None'}: {name} ({int(confidence*100)})% {int(area)}px",
                                color=color.as_bgr(), txt_color=color.text_color().as_bgr())

            original_frame = annotator.result()

    if TARGET_FPS is not None:
        frame_end_time = time.perf_counter()
        frame_time = frame_end_time - frame_start_time
        delay_time = max(1./TARGET_FPS - frame_time, 0)
        time.sleep(delay_time)

    # Write frame to output video if specified
    if video_writer is not None:
        video_writer.write(original_frame)
        if pbar is not None:
            pbar.update(1)

    # Only show window if not saving to file
    if args.output is None:
        cv2.imshow("result", original_frame)
    c = cv2.waitKey(1) if args.output is None else -1
    if c == 107:
        time.sleep(0.1)
        while True:
            c = cv2.waitKey(1)
            if c == 107 or c == 27:
                break

            if args.output is None and cv2.getWindowProperty("result", cv2.WND_PROP_VISIBLE) < 1:
                break

            if c == 114:
                track_history.clear()
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                break

    if c == 27:
        break

    if args.output is None and cv2.getWindowProperty("result", cv2.WND_PROP_VISIBLE) < 1:
        break

    if c == 114:
        track_history.clear()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

cap.release()
if pbar is not None:
    pbar.close()
if video_writer is not None:
    video_writer.release()
    print(f"\nOutput video saved to {args.output}")
if args.output is None:
    cv2.destroyAllWindows()

if args.output is None:
    print(f"Avg FPS: {total_frames / (time.perf_counter() - prog_start)}")

# AVG FPS: 21.545
