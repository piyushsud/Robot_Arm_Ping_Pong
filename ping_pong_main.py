import numpy as np
import cv2
from inverseKinematics import InverseKinematics
from colorChecker import ColorChecker
from ballDetector import BallDetector
from blobDetector import BlobDetector
from frameConverter import FrameConverter
from publisher import MqttPublisher
import pyrealsense2 as rs
import time
from trajectoryCalculator import TrajectoryCalculator
import matplotlib.pyplot as plt
import time

test_path = "C:/Users/piyus/Robot_Arm_Ping_Pong/misc/test_pics/"
trajectory_path = "C:/Users/piyus/Robot_Arm_Ping_Pong/misc/trajectory_pics/"

BLACK_CAMERA_IMAGE_DIM = (480, 640)
REALSENSE_IMAGE_DIM = (480, 640)

FPS = 30
N_CHANNELS = 3
MOTION_THRESHOLD = 100
MAX_INTENSITY = 255
BALL_DEST_X_POSITION = 2 # in meters
TRAJECTORY_N_FRAMES = 2
MIN_VEL = 10 # in pixels/frame in the horizontal direction
NEURAL_NETWORK_IMAGE_SIZE = 96

# camera 1 is the intel realsense, camera 2 is the black camera

class PingPongPipeline:

    def __init__(self):
        # print("starting init function")
        self.invKin = InverseKinematics()
        self.ballDetector = BallDetector()
        self.blobDetector = BlobDetector()
        self.colorChecker = ColorChecker()
        self.frameConverter = FrameConverter()
        self.publisher = MqttPublisher()
        self.trajectoryCalculator = TrajectoryCalculator()

        # intel realsense
        self.realsense = cv2.VideoCapture(3, cv2.CAP_DSHOW)
        self.realsense.set(cv2.CAP_PROP_FRAME_WIDTH, REALSENSE_IMAGE_DIM[1])
        self.realsense.set(cv2.CAP_PROP_FRAME_HEIGHT, REALSENSE_IMAGE_DIM[0])\

        # black camera
        self.black_cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        self.black_cam.set(cv2.CAP_PROP_FRAME_WIDTH, BLACK_CAMERA_IMAGE_DIM[1]*2)
        self.black_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, BLACK_CAMERA_IMAGE_DIM[0])

        self.dist_thresh_pixels = 100
        self.prev_point_realsense = None
        self.prev_point_black = None

    def go_to_ball(self):

        # previous information used for motion tracking
        prev_gray_img_realsense = None
        prev_gray_img_black_camera = None
        prev_time = None
        prev_closest_pt = None

        # other variables
        closest_pt = None
        curr_time = None
        trajectory_frame_count = 0
        ball_dest_estimates = []
        done = False
        final_ball_dest_estimate = (0, 0, 0)  # (x, y, z) in meters
        cannot_reach_ball = False
        i = 0

        # Streaming loop
        try:
            while True:
                # get black camera frame
                ret_black, black_camera_frame = self.black_cam.read()

                # get realsense frame
                ret_realsense, color_image_realsense = self.realsense.read()

                if color_image_realsense is None or black_camera_frame is None:
                    continue

                # get left image from stereo pair of black camera and convert to grayscale
                color_image_black_camera = black_camera_frame[:, 0:640, :]
                gray_image_black_camera = cv2.cvtColor(color_image_black_camera, cv2.COLOR_BGR2GRAY)

                # convert realsense color image to grayscale
                gray_image_realsense = cv2.cvtColor(color_image_realsense, cv2.COLOR_BGR2GRAY)

                # # display camera feeds
                # cv2.imshow("black camera img", gray_image_black_camera)
                # cv2.imshow("realsense img", gray_image_realsense)
                #
                # # the 'q' button is set as the
                # # quitting button
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

                # increment loop counter
                i += 1

                # takes a bit for the exposure to stabilize
                if i < 20:
                    continue

                if prev_gray_img_realsense is not None:

                    ball_detected, point_realsense = self.find_ball(prev_gray_img_realsense,
                                                                    gray_image_realsense,
                                                                    color_image_realsense,
                                                                    "realsense")

                    ball_detected_2, point_black = self.find_ball(prev_gray_img_black_camera,
                                                                  gray_image_black_camera,
                                                                  color_image_black_camera,
                                                                  "black")

                    if ball_detected and ball_detected_2:

                        if self.prev_point_black is not None and self.prev_point_realsense is not None:

                            ball_horizontal_speed = point_realsense[0] - self.prev_point_realsense[0]

                            # if ball has started to move towards the robot
                            if ball_horizontal_speed > MIN_VEL and trajectory_frame_count < TRAJECTORY_N_FRAMES:

                                # position of ball in realsense camera frame, as estimated with realsense
                                x1_d, y1_d, z1_d = self.frameConverter.image_to_camera_frame(
                                    "realsense", point_realsense[0], point_realsense[1])

                                # position of ball in black camera frame, as estimated with black camera
                                x2_e, y2_e, z2_e = self.frameConverter.image_to_camera_frame(
                                    "black", point_black[0], point_black[1])

                                # convert both positions to robot arm frame:
                                x1_c, y1_c, z1_c = self.frameConverter.camera_to_robot_frame("realsense", x1_d, y1_d, z1_d)
                                x2_c, y2_c, z2_c = self.frameConverter.camera_to_robot_frame("black", x2_e, y2_e, z2_e)

                                closest_pt = self.frameConverter.find_intersection_point(x1_c, y1_c, z1_c, x2_c, y2_c, z2_c)

                                if closest_pt is not None:
                                    # print(time.perf_counter())
                                    prev_time = curr_time
                                    curr_time = time.perf_counter()

                                    if closest_pt[0] < 0.2:
                                        print("was not able to calculate trajectory in time")
                                        cannot_reach_ball = True
                                        break

                                    if prev_closest_pt is not None:
                                        ball_dest_at_x = \
                                            self.trajectoryCalculator.calculate_trajectory(
                                                np.array([closest_pt[0], closest_pt[1], closest_pt[2]]),
                                                np.array([prev_closest_pt[0], prev_closest_pt[1], prev_closest_pt[2]]),
                                                curr_time - prev_time,
                                                0.1)
                                        if ball_dest_at_x is not None:
                                            ball_dest_estimates.append(ball_dest_at_x)
                                            trajectory_frame_count += 1
                                        else:
                                            cannot_reach_ball = True
                                            break

                                    prev_closest_pt = closest_pt

                                if trajectory_frame_count == TRAJECTORY_N_FRAMES:
                                    break

                        self.prev_point_realsense = point_realsense
                        self.prev_point_black = point_black

                prev_gray_img_realsense = gray_image_realsense
                prev_gray_img_black_camera = gray_image_black_camera

            if cannot_reach_ball is False:
                # after collecting all the data, find the average estimate of the trajectory of the ball
                ball_dest_estimates = np.array(ball_dest_estimates)
                avg_dest = np.zeros((3, ))

                for i in range(3):
                    avg_dest[i] = np.sum(ball_dest_estimates[:, i])/TRAJECTORY_N_FRAMES

                target_not_reachable, angles = self.invKin.analytical_inverse_kinematics(avg_dest[0], avg_dest[1], avg_dest[2], 0)
                print(avg_dest[0], avg_dest[1], avg_dest[2])
            else:
                target_not_reachable = True

            if target_not_reachable:
                print("target not reachable")
            else:
                # self.publisher.publish_angles(angles)
                print("target reachable")

        finally:
            print("done")
            # self.pipeline.stop()

    def find_ball(self, prev_img_gray, curr_img_gray, curr_img_color, camera):

        img_height = curr_img_color.shape[0]
        img_width = curr_img_color.shape[1]

        # use difference between frames for motion tracking
        frameDelta = cv2.absdiff(prev_img_gray, curr_img_gray)
        motion_img = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

        # remove salt and pepper noise
        motion_img = cv2.medianBlur(motion_img, 17)

        contours, hierarchy = cv2.findContours(motion_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_confidence = 0
        ball_location = None

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)

            left_x = int(center_x - NEURAL_NETWORK_IMAGE_SIZE*2)
            right_x = int(center_x + NEURAL_NETWORK_IMAGE_SIZE*2)
            up_y = int(center_y - NEURAL_NETWORK_IMAGE_SIZE*2)
            down_y = int(center_y + NEURAL_NETWORK_IMAGE_SIZE*2)

            if left_x < 0:
                left_x = 0
                right_x = NEURAL_NETWORK_IMAGE_SIZE * 4
            if up_y < 0:
                up_y = 0
                down_y = NEURAL_NETWORK_IMAGE_SIZE * 4
            if right_x > img_width:
                left_x = img_width - NEURAL_NETWORK_IMAGE_SIZE * 4
                right_x = img_width
            if down_y > img_height:
                up_y = img_height - NEURAL_NETWORK_IMAGE_SIZE * 4
                down_y = img_height

            cropped_color_image = curr_img_color[up_y:down_y, left_x:right_x]

            # input image to this function is 384 x 384
            ball_detected, xBox, yBox, wBox, hBox, confidence = self.ballDetector.find_ball_bbox(cropped_color_image, left_x, up_y)
            blob_detected, x_cen, y_cen = self.blobDetector.find_ball(cropped_color_image)

            if ball_detected:
                if confidence > max_confidence:
                    valid = self.colorChecker.is_valid(curr_img_color[yBox:(yBox + hBox), xBox:(xBox + wBox)])
                    if valid:
                        xmax = xBox
                        ymax = yBox
                        wmax = wBox
                        hmax = hBox
                        max_confidence = confidence
                        ball_location = (xmax + wmax/2, ymax + hmax/2)
            elif blob_detected:
                location = (x_cen, y_cen)
                if camera == "realsense":
                    valid = (np.linalg.norm(location - self.prev_point_realsense) < self.dist_thresh_pixels)
                if camera == "black":
                    valid = (np.linalg.norm(location - self.prev_point_black) < self.dist_thresh_pixels)
                if valid:
                    ball_location = location

        if ball_location is None:  # if no bounding boxes or blobs were detected
            return False, None
        else:
            return True, ball_location

if __name__ == "__main__":
    pipeline = PingPongPipeline()
    pipeline.go_to_ball()