import cv2

# 1. Define markers and their corresponding custom messages
MARKER_MESSAGES = {
    1: "Action Alpha: Marker 1 detected!",
    2: "Action Bravo: Marker 2 detected!",
    3: "Action Charlie: Marker 3 detected!",
    4: "Action Delta: Marker 4 detected!"
}

# 2. Initialize the ArUco detector using the modern OpenCV API
# We use the standard 6X6 family (contains 250 possible IDs)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# 3. Start the webcam video capture (0 is usually the default built-in webcam)
camera = cv2.VideoCapture(2)

print("Starting scanner")

while True:
    success, frame = camera.read()
    if not success:
        print("Failed to grab frame.")
        break

    # Convert to grayscale (ArUco detection works better/faster in gray)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the markers
    corners, ids, rejected = detector.detectMarkers(gray)

    # If at least one marker is found
    if ids is not None:
        # Draw outlines and IDs on the frame for visual feedback
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        # Loop through all detected IDs in the current frame
        for marker_id in ids.flatten():
            # Check if the detected ID matches our target dictionary
            if marker_id in MARKER_MESSAGES:
                print(f"[FOUND] {MARKER_MESSAGES[marker_id]}")
            else:
                print(f"[UNKNOWN] Found marker ID {marker_id}, but no message is assigned to it.")

    # Display the live webcam view
    cv2.imshow("ArUco Scanner", frame)

    cv2.waitKey(1)