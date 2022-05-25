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

n_valid_images = 0
for i in range(185):
    image = cv2.imread("C:/Users/piyus/Robot_Arm_Ping_Pong/camera_calibration/realsense_images/checkerboard_" +
                       str(i) + ".png")
    grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    ret, corners = cv2.findChessboardCorners(
                    grayColor, CHECKERBOARD,
                    cv2.CALIB_CB_ADAPTIVE_THRESH
                    + cv2.CALIB_CB_FAST_CHECK +
                    cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret is True:
        threedpoints.append(objectp3d)
        corners2 = cv2.cornerSubPix(
                grayColor, corners, (11, 11), (-1, -1), criteria)
        twodpoints.append(corners2)
        n_valid_images += 1
    print(i)

print(n_valid_images)

# Perform camera calibration by
# passing the value of above found out 3D points (threedpoints)
# and its corresponding pixel coordinates of the
# detected corners (twodpoints)
ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
    threedpoints, twodpoints, grayColor.shape[::-1], None, None)



# cv2.solvePnp(objectp3d, corners3, matrix, distortion)

# Displaying required output
print(" Camera matrix:")
print(matrix)

print("\n Distortion coefficient:")
print(distortion)

print("\n Rotation Vectors:")
print(r_vecs)

print("\n Translation Vectors:")
print(t_vecs)

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


mean_error = 0
for i in range(len(threedpoints)):
    imgpoints2, _ = cv2.projectPoints(threedpoints[i], r_vecs[i], t_vecs[i], matrix, distortion)
    error = cv2.norm(twodpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(threedpoints)) )