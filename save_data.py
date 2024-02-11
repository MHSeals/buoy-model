import cv2
import time
import numpy as np
import os

PATH = "nav_channel"

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

USE_REALSENSE = True

DISP_SIZE = (1280, 720)

TARGET_FPS = 2

if USE_REALSENSE:
    cap = RealsenseCapture()

else:
    # can change to use different webcams
    cap = cv2.VideoCapture("video.mp4")

if not cap.isOpened():
    raise IOError("Cannot open webcam")

if not os.path.exists(PATH):
    os.mkdir(PATH)

while True:
    frame_start_time = time.perf_counter()
    frame = cap.read()[1]

    if TARGET_FPS is not None:
        frame_end_time = time.perf_counter()
        frame_time = frame_end_time - frame_start_time
        delay_time = max(1./TARGET_FPS - frame_time, 0)
        time.sleep(delay_time)

    cv2.imwrite(f"{PATH}/{time.time()}.jpg", frame)

    frame = cv2.resize(frame, DISP_SIZE)
    cv2.imshow("result", frame)
    c = cv2.waitKey(1)
    if c == 107:
        time.sleep(0.1)
        while True:
            c = cv2.waitKey(1)
            if c == 107 or c == 27:
                break

            if cv2.getWindowProperty("result", cv2.WND_PROP_VISIBLE) < 1:
                break

            if c == 114:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                break

    if c == 27:
        break

    if cv2.getWindowProperty("result", cv2.WND_PROP_VISIBLE) < 1:
        break

    if c == 114:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

cap.release()
cv2.destroyAllWindows()