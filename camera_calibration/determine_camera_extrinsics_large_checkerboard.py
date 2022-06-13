import numpy as np
import cv2

# Define the dimensions of checkerboard
CHECKERBOARD = (6, 9)
# CHECKERBOARD = (4, 7)

# stop the iteration when specified
# accuracy, epsilon, is reached or
# specified number of iterations are completed.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 3D points real world coordinates
objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
for i in range(9):
    for j in range(6):
        objectp3d[0][i*6 + j] = np.array([8 - i, j, 0])

# for i in range(7):
#     for j in range(4):
#         objectp3d[0][i*4 + j] = np.array([6 - i, j, 0])

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
image_copy = np.copy(image)

image = image[200:430, 150:460]
grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grayColorFull = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

ret, corners = cv2.findChessboardCorners(
    grayColor, CHECKERBOARD,
    cv2.CALIB_CB_ADAPTIVE_THRESH
    + cv2.CALIB_CB_FAST_CHECK +
    cv2.CALIB_CB_NORMALIZE_IMAGE)


corners_full = []
for corner in corners:
    # print(corner[])
    corners_full.append([[corner[0][0] + 150, corner[0][1] + 200]])

corners_full = np.array(corners_full, dtype=np.float32)
# print(corners_full.dtype)
image = cv2.circle(image_copy,  (216, 239),  10, (0, 0, 255), -1)
cv2.imshow("image", image)
cv2.waitKey(0)

if ret is True:
    corners2 = cv2.cornerSubPix(
        grayColorFull, corners_full, (11, 11), (-1, -1), criteria)

    # image_copy = cv2.drawChessboardCorners(image_copy, CHECKERBOARD, corners2, True)
    # cv2.imshow("img", image_copy)
    # cv2.waitKey(0)
    # MAKE SURE ORDER OF 2D POINTS MATCHES ORDER OF 3D POINTS
    print(corners2, objectp3d)

    ret, rvec, tvec = cv2.solvePnP(objectp3d, corners2, camera_matrix, distortion)

    print(rvec, tvec)
else:
    print("failed")
