#camera input
import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not detect a camera.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read from the camera.")
        break

    cv2.imshow("Camera", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()