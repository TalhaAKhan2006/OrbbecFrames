import os
import shutil
import time

import cv2
import numpy as np
from pyorbbecsdk import *


# =========================
# Configuration
# =========================

OUTPUT_DIR = "frames"

CAPTURE_TIME = 10          # seconds
TARGET_FPS = 10             # frames per second
TOTAL_FRAMES = CAPTURE_TIME * TARGET_FPS

COLOR_WIDTH = 1280
COLOR_HEIGHT = 720
COLOR_FPS = 30


# =========================
# Prepare output directory
# =========================

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR)

print(f"Output directory: {OUTPUT_DIR}")
print(f"Capturing {TOTAL_FRAMES} frames at {TARGET_FPS} FPS...")
print()


# =========================
# Initialize camera
# =========================

pipeline = Pipeline()
config = Config()

# Enable color stream
color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)

color_profile = color_profiles.get_video_stream_profile(
    COLOR_WIDTH,
    COLOR_HEIGHT,
    OBFormat.RGB,
    COLOR_FPS
)

config.enable_stream(color_profile)

pipeline.start(config)

print("Camera started.")
print("Beginning capture...")
print()


# =========================
# Capture frames
# =========================

frame_interval = 1.0 / TARGET_FPS

next_capture_time = time.perf_counter()

captured = 0

try:

    while captured < TOTAL_FRAMES:

        # Wait for a new frame
        frames = pipeline.wait_for_frames(1000)

        if frames is None:
            continue

        color_frame = frames.get_color_frame()

        if color_frame is None:
            continue

        # Convert Orbbec frame to NumPy array
        data = np.asarray(color_frame.get_data())

        # RGB -> BGR for OpenCV
        image = cv2.cvtColor(
            data,
            cv2.COLOR_RGB2BGR
        )

        # Save frame
        filename = os.path.join(
            OUTPUT_DIR,
            f"frame_{captured + 1:04d}.jpg"
        )

        cv2.imwrite(filename, image)

        captured += 1

        print(
            f"Captured {captured:03d}/{TOTAL_FRAMES}: "
            f"{filename}"
        )

        # Maintain approximately 10 FPS
        next_capture_time += frame_interval

        sleep_time = next_capture_time - time.perf_counter()

        if sleep_time > 0:
            time.sleep(sleep_time)

finally:

    pipeline.stop()

    print()
    print("Capture complete.")
    print(f"Saved {captured} frames to '{OUTPUT_DIR}/'")