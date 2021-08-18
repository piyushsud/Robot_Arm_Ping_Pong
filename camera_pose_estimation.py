import cv2
import numpy as np

IMG_HEIGHT = 480
IMG_WIDTH = 640
FX = 605.00622559  # focal length in x-direction in pixels
FY = 604.44366455  # focal length in y-direction in pixels
CX = 315.97814941  # optical center x-coordinate in pixels
CY = 246.10261536  # optical center y-coordinate in pixels
# DX = ?        # x displacement between camera and world frame, in terms of world frame, in meters
# DY = ?       # y displacement between camera and world frame, in terms of world frame, in meters
# DZ = ?           # z displacement between camera and world frame, in terms of world frame, in meters
X_FRAME_OFFSET = -0.3
Z_FRAME_OFFSET = -0.2

cameraMatrix = np.array([
    [FX, 0, CX],
    [0, FY, CY],
    [0, 0, 1]
])

distCoeffs = np.array([0, 0, 0, 0, 0])

imagePoints = np.array([
    (),
    (),
    (),
    (),
    (),
    (),
    (),
    (),
    (),
    (),
    (),
    (),
    (),
    ()
], dtype=np.float64)

objectPoints = np.zeros((14, 3))
for i in range(14):
    objectPoints[i, 0] = X_FRAME_OFFSET + 0.01*i
    objectPoints[i, 1] = -2.3 - i*0.25

objectPoints[0, 2] = Z_FRAME_OFFSET - 0.0265
objectPoints[1, 2] = Z_FRAME_OFFSET - 0.0065
objectPoints[2, 2] = Z_FRAME_OFFSET - 0.0265
objectPoints[3, 2] = Z_FRAME_OFFSET - 0.0465
objectPoints[4, 2] = Z_FRAME_OFFSET - 0.0265
objectPoints[5, 2] = Z_FRAME_OFFSET - 0.0065
objectPoints[6, 2] = Z_FRAME_OFFSET - 0.0265
objectPoints[7, 2] = Z_FRAME_OFFSET - 0.0465
objectPoints[8, 2] = Z_FRAME_OFFSET - 0.0265
objectPoints[9, 2] = Z_FRAME_OFFSET - 0.0065
objectPoints[10, 2] = Z_FRAME_OFFSET - 0.0265
objectPoints[11, 2] = Z_FRAME_OFFSET - 0.0465
objectPoints[12, 2] = Z_FRAME_OFFSET - 0.0265
objectPoints[13, 2] = Z_FRAME_OFFSET - 0.0065

cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, flags=0)
