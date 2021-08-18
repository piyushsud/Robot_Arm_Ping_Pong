import numpy as np

IMG_HEIGHT = 480
IMG_WIDTH = 640
FX = 605.00622559  # focal length in x-direction in pixels
FY = 604.44366455  # focal length in y-direction in pixels
CX = 315.97814941  # optical center x-coordinate in pixels
CY = 246.10261536  # optical center y-coordinate in pixels

Tcw = np.array([
    [0.99945658, -0.02437214,  0.02219342, -0.05714963],
    [0.02604821, 0.99656116, -0.07865968, -0.10698736],
    [-0.0202, 0.07919504,  0.99665446, -1.17350852],
    [0., 0., 0., 1.]
])


class FrameConverter:
    def __init__(self):
        pass

    def image_to_camera_frame(self, depth_img, r, c):
        depth = depth_img[c, r]
        x = (c - CX)*depth/FX
        y = (r - CY)*depth/FY
        z = depth
        return x, y, z

    # see conventions.txt for frame orientations
    def camera_to_world_frame(self, x, y, z):

        camera_pt = np.array([x, y, z])
        world_pt = np.matmul(Tcw, camera_pt)

        return world_pt[0], world_pt[1], world_pt[2]

    def world_to_robot_frame(self, x_world, y_world, z_world):

        x_robot = -x_world
        y_robot = -y_world
        z_robot = z_world

        return x_robot, y_robot, z_robot

