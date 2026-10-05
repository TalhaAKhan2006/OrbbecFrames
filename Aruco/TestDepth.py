import cv2

# 1. Define markers and their corresponding custom messages
MARKER_MESSAGES = {
    1: "Action Alpha: Marker 1 detected!",
    2: "Action Bravo: Marker 2 detected!",
    3: "Action Charlie: Marker 3 detected!",
    4: "Action Delta: Marker 4 detected!"
}

# ---------------------------------------------------------
# DEPTH PARAMETERS
# ---------------------------------------------------------

# Actual physical width of your ArUco marker
MARKER_WIDTH_CM = 1.9

# Camera focal length in pixels
# YOU MUST CALIBRATE THIS VALUE FOR YOUR CAMERA
FOCAL_LENGTH_PIXELS = 500

# ---------------------------------------------------------
# ARUCO SETUP
# ---------------------------------------------------------

aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_6X6_250
)

aruco_params = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(
    aruco_dict,
    aruco_params
)

# ---------------------------------------------------------
# CAMERA
# ---------------------------------------------------------

camera = cv2.VideoCapture(0)

print("Starting scanner")

while True:

    success, frame = camera.read()

    if not success:
        print("Failed to grab frame.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(gray)

    # If markers are detected
    if ids is not None:

        # Draw marker outlines
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Process every detected marker
        for i, marker_id in enumerate(ids.flatten()):

            # Get the four corners of this marker
            marker_corners = corners[i][0]

            # -------------------------------------------------
            # CALCULATE MARKER WIDTH
            # -------------------------------------------------

            # Top edge
            top_width = cv2.norm(
                marker_corners[0],
                marker_corners[1]
            )

            # Bottom edge
            bottom_width = cv2.norm(
                marker_corners[3],
                marker_corners[2]
            )

            # Average width
            pixel_width = (top_width + bottom_width) / 2

            # -------------------------------------------------
            # DEPTH CALCULATION
            # -------------------------------------------------

            distance_cm = (
                FOCAL_LENGTH_PIXELS * MARKER_WIDTH_CM
            ) / pixel_width

            # -------------------------------------------------
            # PRINT MARKER INFORMATION
            # -------------------------------------------------

            if marker_id in MARKER_MESSAGES:

                print(
                    f"[FOUND] {MARKER_MESSAGES[marker_id]} "
                    f"| Distance: {distance_cm:.2f} cm"
                )

            else:

                print(
                    f"[UNKNOWN] Marker ID {marker_id} "
                    f"| Distance: {distance_cm:.2f} cm"
                )

            # -------------------------------------------------
            # DISPLAY DISTANCE ON VIDEO
            # -------------------------------------------------

            center_x = int(
                sum(point[0] for point in marker_corners) / 4
            )

            center_y = int(
                sum(point[1] for point in marker_corners) / 4
            )

            text = f"{distance_cm:.1f} cm"

            cv2.putText(
                frame,
                text,
                (center_x - 40, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Display camera
    cv2.imshow("ArUco Scanner", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
