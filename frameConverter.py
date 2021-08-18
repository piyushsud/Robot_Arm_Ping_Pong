import numpy as np

IMG_HEIGHT = 480
IMG_WIDTH = 640
FX = 605.00622559  # focal length in x-direction in pixels
FY = 604.44366455  # focal length in y-direction in pixels
CX = 315.97814941  # optical center x-coordinate in pixels
CY = 246.10261536  # optical center y-coordinate in pixels


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

        #todo: find T
        T = np.zeros((4, 3))

        camera_pt = np.array([x, y, z])
        world_pt = np.matmul(T, camera_pt)

        return world_pt[0], world_pt[1], world_pt[2]
