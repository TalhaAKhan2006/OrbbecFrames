import cv2
import time

camera = cv2.VideoCapture(2)
#camera.set(cv2.CAP_PROP_FPS, 60)
if not camera.isOpened():
    raise RuntimeError("Could not detect a camera.")

# Initialize variables for FPS calculation
prev_time = 0

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read from the camera.")
        break

    # Calculate current FPS
    current_time = time.time()
    # Avoid division by zero on the first frame
    fps = 1 / (current_time - prev_time) if prev_time > 0 else 0 
    prev_time = current_time

    # Draw the FPS on the frame
    fps_text = f"FPS: {int(fps)}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Camera", frame)

    cv2.waitKey(1)

camera.release()
cv2.destroyAllWindows()