import numpy as np
import cv2
from datetime import datetime
from inverse_kinematics import InverseKinematics
from ballDetector import BallDetector
from frameConverter import FrameConverter
from publisher import MqttPublisher

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

    def go_to_ball():
        previous_rgb_img = np.zeros(IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS)
        previous_bbox_center = (0, 0)  # in pixels
        previous_ball_precise_location = (0, 0)  # in pixels
        previous_ball_precise_location_world_frame = (0, 0, 0)  # (x, y, z) in meters
        trajectory_frame_count = 0
        final_ball_dest_estimate = (0, 0, 0)  # (x, y, z) in meters
        ball_dest_estimates = []
        prev_time = None
        while True:
            rgb_img = read_rgb_img()  # 1920 x 1080
            depth_img = read_depth_img()  # 1280 x 720
            rgb_img = cv2.resize(rgb_img, (720, 1280))


            # translate depth frame to rgb frame pyrealsense2.align

            motion_img = rgb_img - previous_rgb_img
            motion_img_intensity = cv2.cvtColor(motion_img, cv2.COLOR_BGR2GRAY)
            ret, thresholded_img = cv2.threshold(motion_img_intensity, MOTION_THRESHOLD, MAX_INTENSITY, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresholded_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                ball_detected, ball_bounding_box = self.ballDetector.find_ball_bbox(rgb_img[y:y+h, x:x+w])
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
    go_to_ball()