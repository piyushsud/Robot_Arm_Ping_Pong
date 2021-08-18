import numpy as np
import cv2
from inverseKinematics import InverseKinematics
from ballDetector import BallDetector
from frameConverter import FrameConverter
from publisher import MqttPublisher
import pyrealsense2 as rs
import time
from trajectoryCalculator import TrajectoryCalculator

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
N_CHANNELS = 3
MOTION_THRESHOLD = 100
MAX_INTENSITY = 255
BALL_DEST_X_POSITION = 2 # in meters
TRAJECTORY_N_FRAMES = 4
MIN_VEL = 10 # in pixels/frame in the horizontal direction

# robot is on the right, player is on the left

class PingPongPipeline:

    def __init__(self):
        self.invKin = InverseKinematics()
        self.ballDetector = BallDetector()
        self.frameConverter = FrameConverter()
        self.publisher = MqttPublisher()
        self.trajectoryCalculator = TrajectoryCalculator()

        self.pipeline = rs.pipeline()

        # Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
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
        self.profile = self.pipeline.start(config)

    def go_to_ball(self):
        # previous_bbox_center = (0, 0)  # in pixels
        # previous_ball_precise_location = (0, 0)  # in pixels
        # previous_ball_precise_location_world_frame = (0, 0, 0)  # (x, y, z) in meters
        # final_ball_dest_estimate = (0, 0, 0)  # (x, y, z) in meters
        prev_color_img = None
        prev_time = None
        prev_bbox_center = None
        trajectory_frame_count = 0
        prev_x_world = None
        prev_y_world = None
        prev_z_world = None
        ball_dest_estimates = []
        dest_x_avg = None
        dest_y_avg = None
        dest_z_avg = None

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = self.profile.get_device().first_depth_sensor()

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        align = rs.align(align_to)
        i = 0
        # Streaming loop
        try:
            while True:

                # Get frameset of color and depth
                frames = self.pipeline.wait_for_frames()

                # takes a bit for the exposure to stabilize
                if i < 20:
                    i += 1
                    continue

                # Align the depth frame to color frame
                aligned_frames = align.process(frames)

                # Get aligned frames
                aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
                color_frame = aligned_frames.get_color_frame()

                # Validate that both frames are valid
                if not aligned_depth_frame or not color_frame:
                    continue

                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                gray_image = cv2.GaussianBlur(gray_image, (21, 21), 0)

                if prev_color_img is not None:
                    prev_gray_img = cv2.cvtColor(prev_color_img, cv2.COLOR_BGR2GRAY)
                    frameDelta = cv2.absdiff(prev_gray_img, gray_image)
                    motion_img = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

                    contours, hierarchy = cv2.findContours(motion_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        ball_detected, ball_bounding_box = self.ballDetector.find_ball_bbox(color_image[y:y+h, x:x+w], x, y)
                        if ball_detected:
                            # find coordinates of center of ball
                            r = int(y + h/2)
                            c = int(x + w/2)
                            bbox_center = (r, c)

                            if prev_bbox_center is not None:
                                # velocity in units of pixels per frame
                                ball_horizontal_velocity = bbox_center[1] - prev_bbox_center[1]
                                if ball_horizontal_velocity > MIN_VEL and trajectory_frame_count < TRAJECTORY_N_FRAMES:
                                    x, y, z = self.frameConverter.image_to_camera_frame(depth_image, r, c)
                                    x_world, y_world, z_world = self.frameConverter.camera_to_world_frame(x, y, z)

                                    curr_time = time.time()
                                    ball_dest_at_x = \
                                        self.trajectoryCalculator.calculate_trajectory(x_world, y_world, z_world,
                                                                                       prev_x_world, prev_y_world,
                                                                                       prev_z_world, curr_time - prev_time)
                                    ball_dest_estimates.append(ball_dest_at_x)
                                    trajectory_frame_count += 1
                                if trajectory_frame_count == TRAJECTORY_N_FRAMES:
                                    x_sum = 0
                                    y_sum = 0
                                    z_sum = 0

                                    for estimate in ball_dest_estimates:
                                        x_sum += estimate[0]
                                        y_sum += estimate[1]
                                        z_sum += estimate[2]

                                    dest_x_avg = x_sum / TRAJECTORY_N_FRAMES
                                    dest_y_avg = y_sum / TRAJECTORY_N_FRAMES
                                    dest_z_avg = z_sum / TRAJECTORY_N_FRAMES
                                    break
                else:
                    continue
                break
            target_reachable, angles = self.invKin.analytical_inverse_kinematics(dest_x_avg, dest_y_avg, dest_z_avg, 0)
            if target_reachable:
                self.publisher.publish_angles(angles)
            else:
                print("target not reachable")

        finally:
            self.pipeline.stop()

if __name__ == "__main__":
    pipeline = PingPongPipeline()
    pipeline.go_to_ball()