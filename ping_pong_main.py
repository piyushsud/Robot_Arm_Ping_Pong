import numpy as np
import cv2
from datetime import datetime
from inverse_kinematics import InverseKinematics
from ballDetector import BallDetector
from frameConverter import FrameConverter
from publisher import MqttPublisher
import pyrealsense2 as rs

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
N_CHANNELS = 3
MOTION_THRESHOLD = 100
MAX_INTENSITY = 255
BALL_DEST_X_POSITION = 2 # in meters
TRAJECTORY_N_FRAMES = 4
MIN_VEL_TRAJECTORY_EST_LEFT = -10 # in pixels/frame in the horizontal direction

# robot is on the left, player is on the right

class PingPongPipeline:

    def __init__(self):
        self.invKin = InverseKinematics()
        self.ballDetector = BallDetector()
        self.frameConverter = FrameConverter()
        self.publisher = MqttPublisher()

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
        prev_color_img = None
        # previous_bbox_center = (0, 0)  # in pixels
        # previous_ball_precise_location = (0, 0)  # in pixels
        # previous_ball_precise_location_world_frame = (0, 0, 0)  # (x, y, z) in meters
        # trajectory_frame_count = 0
        # final_ball_dest_estimate = (0, 0, 0)  # (x, y, z) in meters
        # ball_dest_estimates = []
        prev_time = None


        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        # print("Depth Scale is: ", depth_scale)

        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away
        clipping_distance_in_meters = 1  # 1 meter
        clipping_distance = clipping_distance_in_meters / depth_scale

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
                        if ball_horizontal_velocity < MIN_VEL_TRAJECTORY_EST_LEFT and trajectory_frame_count < TRAJECTORY_N_FRAMES:
                            ball_precise_location = find_ball_precise_location(previous_bbox_center, ball_bounding_box)
                            ball_precise_location_world_frame = pixel_to_world_frame(left_img, depth_img, ball_precise_location)
                            curr_time = datetime.now()
                            ball_dest_at_x = calculate_trajectory(ball_precise_location_world_frame, previous_ball_precise_location_world_frame, curr_time - prev_time)
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

                            x_avg = x_sum / TRAJECTORY_N_FRAMES
                            y_avg = y_sum / TRAJECTORY_N_FRAMES
                            z_avg = z_sum / TRAJECTORY_N_FRAMES
                            final_ball_dest_estimate = (x_avg, y_avg, z_avg)
                            break
                else:
                    continue
                break
            angles = inverse_kinematics(final_ball_dest_estimate)
            send_mqtt(angles)

if __name__ == "__main__":
    pipeline = PingPongPipeline()
    pipeline.go_to_ball()