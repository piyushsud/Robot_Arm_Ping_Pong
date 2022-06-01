import cv2
import numpy as np


SQUARE_SIZE = 0.045  # in meters

# checkerboard upright frame = frame A
# checkerboard on table frame = frame B
# robot arm frame = frame C
# realsense frame = frame D
# black camera frame = frame E

# black camera parameters:

black_camera_matrix = np.array([
    [1.15090805e+03, 0.00000000e+00, 6.85823395e+02],
    [0.00000000e+00, 1.14906092e+03, 4.79730926e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

black_camera_distortion = np.array([[ 0.06571678, -0.06531794, -0.00267922, -0.00469088, -0.05419306]])

# intel realsense parameters:

realsense_camera_matrix = np.array([
    [591.48522865, 0., 322.52619954],
    [0., 591.9638662, 257.7496392],
    [0., 0., 1.]])

realsense_distortion = np.array([[ 0.00252699,  0.50539056,  0.00564689,  0.00742319, -1.62753842]])


T_ad = np.array([[0.99929197, 0.02608819, 0.02711012, -0.06647676345],
                 [-0.02659131, 0.99947765, 0.01836679, -0.07870460175],
                 [-0.0266168, -0.01907468,  0.99946371, 0.9110652129],
                 [0, 0, 0, 1]])


T_be = np.array([[0.99965129, -0.01992817, -0.01732549, 0.1424825505],
                 [0.01924965,  0.99907374, -0.03848515, -0.14332873005],
                 [0.01807638,  0.03813822,  0.99910896, 1.0483714251],
                 [0, 0, 0, 1]])

T_ca = np.array(
    [
        [-1, 0, 0, 0.4534875],
        [0, 0, -1, -0.2311625],
        [0, -1, 0, 0.2],
        [0, 0, 0, 1]
    ]
)

T_cb = np.array(
    [
        [1, 0, 0, 0.4525],
        [0, -1, 0, 0.0545],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ]
)


class FrameConverter:

    def __init__(self):
        
        # intel realsense intrinsic parameters
        self.realsense_fx = realsense_camera_matrix[0][0]  # focal length in x-direction in pixels
        self.realsense_fy = realsense_camera_matrix[1][1]  # focal length in y-direction in pixels
        self.realsense_cx = realsense_camera_matrix[0][2]  # optical center x-coordinate in pixels
        self.realsense_cy = realsense_camera_matrix[1][2]  # optical center y-coordinate in pixels

        # black camera intrinsic parameters
        self.black_fx = black_camera_matrix[0][0]  # focal length in x-direction in pixels
        self.black_fy = black_camera_matrix[1][1]  # focal length in y-direction in pixels
        self.black_cx = black_camera_matrix[0][2]  # optical center x-coordinate in pixels
        self.black_cy = black_camera_matrix[1][2]  # optical center y-coordinate in pixels

    def image_to_robot_frame(self, camera, r, c):
        x, y, z = self.image_to_camera_frame(camera, r, c)
        point = self.camera_to_robot_frame(camera, x, y, z)
        return point

    def image_to_camera_frame(self, camera, r, c):
        depth = 0  # I am not using the depth estimation from either of the cameras

        if camera == "realsense":
            fx = self.realsense_fx
            fy = self.realsense_fy
            cx = self.realsense_cx
            cy = self.realsense_cy

        else:  # if black camera
            fx = self.black_fx
            fy = self.black_fy
            cx = self.black_cx
            cy = self.black_cy
            
        x = (c - cx)*depth/fx
        y = (r - cy)*depth/fy
        z = depth

        return x, y, z

    def camera_to_robot_frame(self, camera, x, y, z):

        point = np.array([x, y, z, 1])

        if camera == "realsense":
            p_a = np.matmul(T_ad, point)
            p_c = np.matmul(T_ca, p_a)

        else:  # if black camera
            p_b = np.matmul(T_be, point)
            p_c = np.matmul(T_cb, p_b)

        return p_c[:-1]  # point is (x, y, z, 1) so just return (x, y, z)
