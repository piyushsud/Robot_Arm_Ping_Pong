import numpy as np
import cv2

# Define the dimensions of checkerboard
# CHECKERBOARD = (9, 6)
CHECKERBOARD = (4, 7)

# stop the iteration when specified
# accuracy, epsilon, is reached or
# specified number of iterations are completed.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 3D points real world coordinates
objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
# for i in range(6):
#     for j in range(9):
#         objectp3d[0][i*9 + j] = np.array([i, 8 - j, 0])

for i in range(7):
    for j in range(4):
        objectp3d[0][i*4 + j] = np.array([6 - i, j, 0])

camera_matrix = np.array([[591.48522865,   0.,         322.52619954],
 [  0.,         591.9638662,  257.7496392 ],
 [  0.,           0.,           1.        ]])

distortion = np.array([[ 0.00252699,  0.50539056,  0.00564689,  0.00742319, -1.62753842]])

# camera_matrix = np.array([[1.15090805e+03, 0.00000000e+00, 6.85823395e+02],
#  [0.00000000e+00, 1.14906092e+03, 4.79730926e+02],
#  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
#
# distortion = np.array([[ 0.06571678, -0.06531794, -0.00267922, -0.00469088, -0.05419306]])

image = cv2.imread("C:/Users/piyus/Robot_Arm_Ping_Pong/camera_calibration/realsense_images/checkerboard_upright.png")

grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, corners = cv2.findChessboardCorners(
    grayColor, CHECKERBOARD,
    cv2.CALIB_CB_ADAPTIVE_THRESH
    + cv2.CALIB_CB_FAST_CHECK +
    cv2.CALIB_CB_NORMALIZE_IMAGE)

image = cv2.circle(image,  (279, 207),  10, (0,0,255), -1)
cv2.imshow("image", image)
cv2.waitKey(0)

if ret is True:
    corners2 = cv2.cornerSubPix(
        grayColor, corners, (11, 11), (-1, -1), criteria)

    # MAKE SURE ORDER OF 2D POINTS MATCHES ORDER OF 3D POINTS
    print(corners2, objectp3d)

    ret, rvec, tvec = cv2.solvePnP(objectp3d, corners2, camera_matrix, distortion)

    print(rvec, tvec)
else:
    print("failed")
