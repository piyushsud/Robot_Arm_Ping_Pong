# Import required modules
import cv2
import numpy as np
import os
import math

# Define the dimensions of checkerboard
CHECKERBOARD = (9, 6)

# stop the iteration when specified
# accuracy, epsilon, is reached or
# specified number of iterations are completed.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Vector for 3D points
threedpoints = []

# Vector for 2D points
twodpoints = []


# 3D points real world coordinates
objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
#print(np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]])
# a = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]]
# b = a.T.reshape(-1, 2)
# objectp3d[0, :, :2] = b
for i in range(6):
    for j in range(9):
        objectp3d[0][i*9 + j] = np.array([i, 8 - j, 0])
# print(objectp3d)


for i in range(226):
    image = cv2.imread("C:/Users/piyus/Robot_Arm_Ping_Pong/camera_calibration/black_camera_images/left/checkerboard_4.png")

#image = cv2.resize(image, (image.shape[0]*4, image.shape[1]*4))
grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Find the chess board corners
# # If desired number of corners are
# # found in the image then ret = true
# ret, corners = cv2.findChessboardCorners(
#                 grayColor, CHECKERBOARD,
#                 cv2.CALIB_CB_ADAPTIVE_THRESH
#                 + cv2.CALIB_CB_FAST_CHECK +
#                 cv2.CALIB_CB_NORMALIZE_IMAGE)
#
# most_left = image.shape[1]
# most_right = 0
# most_up = image.shape[0]
# most_down = 0
#
# corners2 = cv2.cornerSubPix(grayColor, corners, (11, 11), (-1, -1), criteria)
#
# for corner in corners2:
#     x = int(corner[0][0])
#     y = int(corner[0][1])
#     if x < most_left:
#         most_left = x
#     if x > most_right:
#         most_right = x
#     if y < most_up:
#         most_up = y
#     if y > most_down:
#         most_down = y
#
# cropped_img = image[(most_up - 50):(most_down + 50), (most_left - 50):(most_right + 50)]
# resized_cropped_image = cv2.resize(cropped_img, (cropped_img.shape[1], cropped_img.shape[0]))
# grayColor = cv2.cvtColor(resized_cropped_image, cv2.COLOR_BGR2GRAY)

# find corners again now that we have our cropped image

ret, corners = cv2.findChessboardCorners(
                grayColor, CHECKERBOARD,
                cv2.CALIB_CB_ADAPTIVE_THRESH
                + cv2.CALIB_CB_FAST_CHECK +
                cv2.CALIB_CB_NORMALIZE_IMAGE)

if ret is True:
    corners2 = cv2.cornerSubPix(
            grayColor, corners, (11, 11), (-1, -1), criteria)
else:
    print("failed")
# cv2.imshow("cropped image", resized_cropped_image)
# cv2.waitKey(0)

# If desired number of corners can be detected then,
# refine the pixel coordinates and display
# them on the images of checker board

# corners3 = []  # to store the corners in the actual image
# print(ret)
# if ret is True:
#     threedpoints.append(objectp3d)
#
#     # Refining pixel coordinates
#     # for given 2d points.
#     corners2 = cv2.cornerSubPix(
#         grayColor, corners, (11, 11), (-1, -1), criteria)
#
#     twodpoints.append(corners2)
#     # print(corners2)
#
#     for corner in corners2:
#         x = corner[0][0] + most_left - 50
#         y = corner[0][1] + most_up - 50
#         corners3.append([[x, y]])
#
#
# corners3 = np.array(corners3, dtype=np.float32)


#cv2.imshow('img', image)
#cv2.waitKey(0)

# print(corners3, objectp3d)
#h, w = image.shape[:2]


# Perform camera calibration by
# passing the value of above found out 3D points (threedpoints)
# and its corresponding pixel coordinates of the
# detected corners (twodpoints)
ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
	objectp3d, corners2, grayColor.shape[::-1], None, None)

# cv2.solvePnp(objectp3d, corners3, matrix, distortion)

# # Displaying required output
# print(" Camera matrix:")
# print(matrix)
#
# print("\n Distortion coefficient:")
# print(distortion)
#
# print("\n Rotation Vectors:")
# print(r_vecs)
#
# print("\n Translation Vectors:")
# print(t_vecs)

# sum = 0
# for vec in t_vecs[0]:
#     sum += vec**2
#
# print(math.sqrt(sum))


# camera_matrix = np.array([[382.29693604, 0., 319.07116699],
                          # [0., 382.29693604, 239.56895447],
                          # [0., 0., 1.]])
#
# ret, r, t = cv2.solvePnP(objectp3d, corners3, camera_matrix, np.array([0, 0, 0, 0, 0]))
# print(r, t)


# mean_error = 0
# for i in range(len(threedpoints)):
# 	imgpoints2, _ = cv2.projectPoints(threedpoints[i], r_vecs[i], t_vecs[i], matrix, distortion)
# 	error = cv2.norm(twodpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
# 	mean_error += error
# print( "total error: {}".format(mean_error/len(threedpoints)) )