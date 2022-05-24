## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()

# cfg = pipeline.start() # Start pipeline and get the configuration it found
# profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
# intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
#
# camera_matrix = np.array([
#     [intr.fx, 0, intr.ppx],
#     [0, intr.fy, intr.ppy],
#     [0, 0, 1]
# ])
#
# print(camera_matrix)

config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# profile = config.get_stream(rs.stream.depth)  # Fetch stream profile for depth stream
# intr = profile.as_video_stream_profile().get_intrinsics()  # Downcast to video_stream_profile and fetch intrinsics
# print(intr.ppx)

try:
    i = 0
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        i += 1



        if i == 10:
            cv2.imwrite("C:/Users/piyus/Robot_Arm_Ping_Pong/camera_calibration/checkerboard_1.png", color_image)
            break
        #cv2.imshow('RealSense', color_image)
        #cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()