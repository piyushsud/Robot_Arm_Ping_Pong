IMG_HEIGHT = 720
IMG_WIDTH = 1280
FX = 500  # focal length in x-direction in pixels
FY = 500  # focal length in y-direction in pixels
CX = 300  # optical center x-coordinate in pixels
CY = 300  # optical center y-coordinate in pixels
DX = 10   # x displacement between camera and world frame, in terms of world frame
DY = 10   # y displacement between camera and world frame, in terms of world frame
DZ = 0    # z displacement between camera and world frame, in terms of world frame

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

        # rotation
        xWorld = -x
        yWorld = -z
        zWorld = -y

        # translation
        xWorld += DX
        yWorld += DY
        zWorld += DZ

        return xWorld, yWorld, zWorld
