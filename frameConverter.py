import numpy as np

IMG_HEIGHT = 480
IMG_WIDTH = 640
FX = 605.00622559  # focal length in x-direction in pixels
FY = 604.44366455  # focal length in y-direction in pixels
CX = 315.97814941  # optical center x-coordinate in pixels
CY = 246.10261536  # optical center y-coordinate in pixels

# realsense to world transformation matrix
Tcw = np.array([
    [0.99945658, -0.02437214, 0.02219342, -0.05714963],
    [-0.0202, 0.07919504, 0.99665446, -1.17350852],
    [-0.02604821, -0.99656116, 0.07865968, 0.10698736],
    [0., 0., 0., 1.]
])

# camera 2 to world transformation matrix
Tcw2 = np.array([
    [-0.0806599138, 0.0857617880, -0.993045263, 1.39795874],
    [0.996714447, -0.000424886245, -0.0809946368, 0.142971742],
    [-0.00736817614, -0.996315580, -0.0854457418, 0.648028519],
    [0, 0, 0, 1]
])

class FrameConverter:
    def __init__(self):
        pass

    def image_to_camera_frame(self, depth_img, r, c):
        depth = depth_img[c, r]/1000
        x = (c - CX)*depth/FX
        y = (r - CY)*depth/FY
        z = depth
        return x, y, z

    # see conventions.txt for frame orientations
    def camera_to_world_frame(self, x, y, z, camera):
        camera_pt = np.array([x, y, z, 1])

        if camera == "realsense":
            world_pt = np.matmul(Tcw, camera_pt)
        else:
            world_pt = np.matmul(Tcw2, camera_pt)

        return world_pt[0], world_pt[1], world_pt[2]

    def world_to_robot_frame(self, x_world, y_world, z_world):

        x_robot = -x_world
        y_robot = -y_world
        z_robot = z_world

        return x_robot, y_robot, z_robot

