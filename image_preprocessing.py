import cv2
import numpy as np
import time

def preprocess(image: np.ndarray) -> np.ndarray:
    # dynamic range compression
    image = cv2.convertScaleAbs(image, alpha=1.0, beta=50)

    # lower brightness
    image = np.clip(image - 50, 0, 255)

    # increase saturation
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = np.array(image, dtype=np.float64)
    image[:, :, 1] = image[:, :, 1] * 1.4
    image[:, :, 1] = np.clip(image[:, :, 1], 0, 255)
    image = np.array(image, dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    # increase contrast
    image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)

    return image

if __name__ == "__main__":
    cap = cv2.VideoCapture("video.mp4")

    average_preprocess_time = 0
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start = time.perf_counter_ns()
        frame = preprocess(frame)
        end = time.perf_counter_ns()

        average_preprocess_time += end - start
        i += 1

        print(f"Average preprocess time: {(average_preprocess_time / i) / 1e6} ms")

        cv2.imshow("result", frame)

        c = cv2.waitKey(1)

        if c == 27:
            break
        
        if cv2.getWindowProperty("result", cv2.WND_PROP_VISIBLE) < 1:
            break

        time.sleep(1/5)

    cap.release()