import cv2
import numpy as np

IMG_HEIGHT = 480
IMG_WIDTH = 640

# realsense camera params:

# FX = 605.00622559  # focal length in x-direction in pixels
# FY = 604.44366455  # focal length in y-direction in pixels
# CX = 315.97814941  # optical center x-coordinate in pixels
# CY = 246.10261536  # optical center y-coordinate in pixels

# other camera params:

FX = 574.94583505  # focal length in x-direction in pixels
FY = 575.36383871  # focal length in y-direction in pixels
CX = 320.0         # optical center x-coordinate in pixels
CY = 240.0         # optical center y-coordinate in pixels

# DX = ?        # x displacement between camera and world frame, in terms of world frame, in meters
# DY = ?       # y displacement between camera and world frame, in terms of world frame, in meters
# DZ = ?           # z displacement between camera and world frame, in terms of world frame, in meters

SQUARE_WIDTH = 0.0235 # in meters

cameraMatrix = np.array([
    [FX, 0, CX],
    [0, FY, CY],
    [0, 0, 1]
])

# distCoeffs = np.array([0, 0, 0, 0, 0], dtype=np.float64)
distCoeffs = np.array([0.06088478, 0.00711266, -0.00356545, -0.00076046, -0.32904665], dtype=np.float64)

# checkerboard = cv2.imread('C:/Users/piyus/Robot_Arm_Ping_Pong/misc/checkerboard_img.jpg')
checkerboard = cv2.imread('C:/Users/piyus/Robot_Arm_Ping_Pong/misc/checkerboard_image_camera2.jpg')

# cropped_checkerboard = checkerboard[230:380, 130:330]
cropped_checkerboard = checkerboard[380:450, 313:413]


# cv2.imshow("checkerboard", cropped_checkerboard)
# cv2.waitKey(0)
img_gray = cv2.cvtColor(cropped_checkerboard, cv2.COLOR_BGR2GRAY)
frame = cv2.resize(img_gray, (img_gray.shape[1] * 3, img_gray.shape[0] * 3))
# frame = img_gray
found_chessboard, found_corners = cv2.findChessboardCorners(frame, (6, 9), flags=cv2.CALIB_CB_FAST_CHECK +
                                                                cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                                cv2.CALIB_CB_NORMALIZE_IMAGE)

# image = cv2.drawChessboardCorners(cropped_checkerboard,
# 										(6, 9),
# 										found_corners, found_chessboard)
# print(found_chessboard)
# print(found_corners)
# cv2.imshow("cropped checkerboard with corners", image)
# cv2.waitKey(0)

objectPoints = []
imagePoints = []

for i in range(9):
    for j in range(6):
        # other camera
        x = 0
        y = 0.046 + SQUARE_WIDTH * i
        z = 0.0345 + SQUARE_WIDTH * j

        # # realsense
        # x = -0.23923125 + SQUARE_WIDTH * i
        # y = -0.3937
        # z = 0.0345 + SQUARE_WIDTH * j

        objectPoints.append((x, y, z))

for corner in found_corners:
    new_corner = corner[0]

    x = new_corner[0]/3 + 313
    y = new_corner[1]/3 + 380

    # x = new_corner[0] + 130
    # y = new_corner[1] + 230

    imagePoints.append((x, y))

objectPointsArr = np.array(objectPoints, dtype=np.float64)
imagePointsArr = np.array(imagePoints, dtype=np.float64)


# 2d point is (c, r)

retval, rvec, tvec = cv2.solvePnP(objectPointsArr, imagePointsArr, cameraMatrix, distCoeffs, flags=0)
# print(rvec, tvec)
arr = np.array([rvec[0][0], rvec[1][0], rvec[2][0]])
# print(rvec[0][0])
R, jacobian = cv2.Rodrigues(arr)
translation = np.array([tvec[0][0], tvec[1][0], tvec[2][0]])

# pt = np.array([0, 0, 0])
# pt = np.array([-0.23923125, -0.3937, 0.03045])
# pt = np.matmul(R, pt)
# x = pt[0] + tvec[0][0]
# y = pt[1] + tvec[1][0]
# z = pt[2] + tvec[2][0]
# print(x, y, z)

# T = Transformation from world frame to camera frame
# Tinv = Transformation from camera frame to world frame

Tinv = np.zeros((4, 4))
Tinv[3, 3] = 1
Tinv[0:3, 0:3] = np.transpose(R)
Tinv[0:3, 3] = -np.matmul(np.transpose(R), translation)
print(Tinv)

# RVECS AND TVECS ARE RELATIVE TO BOTTOM LEFT CORNER!!

# position of bottom left corner:
# x = 23.923125 cm
# y = 39.37 cm
# z = 3.045 cm
# Square width = 2.35 cm




