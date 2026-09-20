from pathlib import Path
import time

import cv2

RECORD_SECONDS = 10
TARGET_FPS = 10
OUTPUT_DIR = Path(__file__).parent / "frames"

OUTPUT_DIR.mkdir(exist_ok=True)
for frame_path in OUTPUT_DIR.iterdir():
    if frame_path.is_file():
        frame_path.unlink()

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FPS, TARGET_FPS)

if not camera.isOpened():
    raise RuntimeError("Could not detect a camera.")

frame_number = 1
start_time = time.monotonic()
next_frame_time = start_time

try:
    while time.monotonic() - start_time < RECORD_SECONDS:
        success, frame = camera.read()

        if not success:
            print("Could not read from the camera.")
            break

        current_time = time.monotonic()
        if current_time >= next_frame_time:
            output_path = OUTPUT_DIR / f"{frame_number}.jpg"
            if not cv2.imwrite(str(output_path), frame):
                raise RuntimeError(f"Could not save {output_path}.")
            frame_number += 1
            next_frame_time += 1 / TARGET_FPS
finally:
    camera.release()

print(f"Saved {frame_number - 1} frames to {OUTPUT_DIR}.")