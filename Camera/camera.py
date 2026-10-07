import cv2
import time

camera = cv2.VideoCapture(2)

# Initialize ArUco detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# camera.set(cv2.CAP_PROP_FPS, 60)
if not camera.isOpened():
    raise RuntimeError("Could not detect a camera.")

# Initialize variables for FPS calculation
prev_time = 0

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read from the camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for ArUco detection

    corners, ids, rejected = detector.detectMarkers(gray)  # Returns corners, ids, and invalid markers

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)  # Draw detected markers on the frame


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