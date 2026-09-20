import cv2

def get_camera_indices():
    index = 0
    arr = []
    # Test indices from 0 up to 10
    while index < 10:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            arr.append(index)
            cap.release()
        index += 1
    return arr

print("Available camera indices:", get_camera_indices())
