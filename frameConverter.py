import cv2
import numpy as np

DIST_TOLERANCE = 0.1

SQUARE_SIZE = 0.045  # in meters

BLACK_CAMERA_IMAGE_DIM = (480, 640)
REALSENSE_IMAGE_DIM = (1080, 1920)

# checkerboard upright frame = frame A
# checkerboard on table frame = frame B
# robot arm frame = frame C
# realsense frame = frame D
# black camera frame = frame E

# # black camera intrinsic parameters:
#
# black_camera_matrix = np.array([
#     [575.454025, 0.00000000e+00, 320],
#     [0.00000000e+00, 574.53046, 240],
#     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
#
# black_camera_distortion = np.array([[ 0.06571678, -0.06531794, -0.00267922, -0.00469088, -0.05419306]])
#
# # intel realsense intrinsic parameters:
#
# realsense_camera_matrix = np.array([
#     [591.48522865, 0., 320],
#     [0., 591.9638662, 240],
#     [0., 0., 1.]])
#
# realsense_distortion = np.array([[ 0.00252699,  0.50539056,  0.00564689,  0.00742319, -1.62753842]])

black_camera_matrix = np.array([
    [575.454025, 0.00000000e+00, 342.9116975],
    [0.00000000e+00, 574.53046, 239.865463],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

black_camera_distortion = np.array([[ 0.06571678, -0.06531794, -0.00267922, -0.00469088, -0.05419306]])

# intel realsense intrinsic parameters:

realsense_camera_matrix = np.array([
    [591.48522865, 0., 322.52619954],
    [0., 591.9638662, 257.7496392],
    [0., 0., 1.]])

realsense_distortion = np.array([[ 0.00252699,  0.50539056,  0.00564689,  0.00742319, -1.62753842]])

T_ad = np.array([[0.99929197, 0.02608819, 0.02711012, 0.06647676345],
                 [-0.02659131, 0.99947765, 0.01836679, 0.07870460175],
                 [-0.0266168, -0.01907468,  0.99946371, -0.9110652129],
                 [0, 0, 0, 1]])


T_be = np.array([[0.99965129, -0.01992817, -0.01732549, -0.1424825505],
                 [0.01924965,  0.99907374, -0.03848515, 0.14332873005],
                 [0.01807638,  0.03813822,  0.99910896, -1.0483714251],
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

        # coords are scaled by 1/z because z is unknown

        x = (c - cx)/fx
        y = (r - cy)/fy
        z = 1

        return x, y, z

    def camera_to_robot_frame(self, camera, x, y, z):

        point = np.array([x, y, z, 1])

        if camera == "realsense":
            p_a = np.matmul(T_ad, point)
            p_c = np.matmul(T_ca, p_a)
            # print("realsense")
            # print(p_a, p_c)

        else:  # if black camera
            p_b = np.matmul(T_be, point)
            p_c = np.matmul(T_cb, p_b)
            # print("black")
            # print(p_b, p_c)

        return p_c[:-1]  # point is (x, y, z, 1) so just return (x, y, z)

    def find_intersection_point(self, x1_c, y1_c, z1_c, x2_c, y2_c, z2_c):

        # https://math.stackexchange.com/questions/2213165/find-shortest-distance-between-lines-in-3d

        # location of ball in world frame as estimated by realsense
        q = np.array([x1_c, y1_c, z1_c])

        # location of ball in world frame as estimated by black camera
        r = np.array([x2_c, y2_c, z2_c])

        # location of realsense in robot frame:
        T_cd = np.matmul(T_ca, T_ad)
        a = T_cd[0:3, 3]

        # location of black camera in robot frame:
        T_ce = np.matmul(T_cb, T_be)
        c = T_ce[0:3, 3]

        # vectors from camera to ball = ball_pos - camera_pos
        b = q - a
        d = r - c

        # print(a, c, q, r, b, d)

        e = a - c

        # expressions that are computed multiple times
        j = np.dot(b, b)
        k = np.dot(d, d)
        l = np.dot(b, d)
        m = np.dot(d, e)
        n = np.dot(b, e)

        # calculate point at line corresponding to minimum distance
        A = -j * k + l**2
        t = (k*n - m*l)/A
        s = (-j*m + n*l)/A

        closest_dist = np.linalg.norm(e + np.dot(b, t) - np.dot(d, s))
        if closest_dist > DIST_TOLERANCE:
            # print("cameras did not agree on location of ball")
            return None
        else:
            closest_pt_1 = a + np.dot(b, t)
            closest_pt_2 = c + np.dot(d, s)
            # print("point 1: " + str(closest_pt_1))
            # print("point 2: " + str(closest_pt_2))
            closest_pt = np.add(closest_pt_1, closest_pt_2)/2
            return closest_pt


