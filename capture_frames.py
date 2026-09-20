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

CAPTURE_TIME = 10       # seconds
TARGET_FPS = 10         # saved frames per second
TOTAL_FRAMES = CAPTURE_TIME * TARGET_FPS

COLOR_WIDTH = 640
COLOR_HEIGHT = 480
COLOR_FPS = 30


# =========================
# Prepare output directory
# =========================

if os.path.exists(OUTPUT_DIR):
    print(f"Removing previous '{OUTPUT_DIR}' directory...")
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR)

print()
print("Orbbec Gemini 335 Frame Capture")
print("--------------------------------")
print(f"Resolution:     {COLOR_WIDTH}x{COLOR_HEIGHT}")
print(f"Camera FPS:     {COLOR_FPS}")
print(f"Capture FPS:    {TARGET_FPS}")
print(f"Duration:       {CAPTURE_TIME} seconds")
print(f"Total frames:   {TOTAL_FRAMES}")
print(f"Output folder:  {OUTPUT_DIR}")
print()


# =========================
# Initialize camera
# =========================

pipeline = Pipeline()
config = Config()

color_profiles = pipeline.get_stream_profile_list(
    OBSensorType.COLOR_SENSOR
)

color_profile = color_profiles.get_video_stream_profile(
    COLOR_WIDTH,
    COLOR_HEIGHT,
    OBFormat.RGB,
    COLOR_FPS
)

config.enable_stream(color_profile)

print("Starting camera...")

pipeline.start(config)

print("Camera started.")
print()
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

        # Wait for a frame from the camera
        frames = pipeline.wait_for_frames(1000)

        if frames is None:
            continue

        color_frame = frames.get_color_frame()

        if color_frame is None:
            continue

        # Convert camera data to NumPy array
        data = np.asarray(color_frame.get_data())

        # Convert RGB to BGR for OpenCV
        image = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

        # Generate filename
        filename = os.path.join(
            OUTPUT_DIR,
            f"frame_{captured + 1:04d}.jpg"
        )

        # Save image
        success = cv2.imwrite(filename, image)

        if not success:
            print(f"ERROR: Could not save {filename}")
            continue

        captured += 1

        print(
            f"Captured {captured:03d}/{TOTAL_FRAMES} "
            f"-> {filename}"
        )

        # Schedule next capture
        next_capture_time += frame_interval

        sleep_time = next_capture_time - time.perf_counter()

        if sleep_time > 0:
            time.sleep(sleep_time)

finally:

    pipeline.stop()

    print()
    print("Camera stopped.")
    print(f"Capture complete: {captured} frames saved.")
    print(f"Images are located in: {OUTPUT_DIR}/")