import cv2
import numpy
import glob
from pathlib import Path

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = numpy.zeros((9*6,3), numpy.float32)
objp[:,:2] = numpy.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(str(Path(__file__).parent / "rs_calibration_images" / "*.jpg"))

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9,6), corners2, ret)
    cv2.imshow('img', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

# PART 2
img = cv2.imread(str(Path(__file__).parent / "rs_calibration_images" / "img2.jpg"))
h, w = img.shape[:2]

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# Undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# Crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

cv2.imshow("calibresult.png", dst)
cv2.waitKey(0)

print("Camera matrix:\n", mtx)
print("dist: \n", dist)         # Distortion coefficients
print("rvecs: \n", rvecs)       # Rotation vectors
print("tvecs: \n", tvecs)       # Translation vectors

cv2.destroyAllWindows()
