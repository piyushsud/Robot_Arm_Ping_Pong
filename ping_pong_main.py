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
TRAJECTORY_N_FRAMES = 3
MIN_VEL = 10 # in pixels/frame in the horizontal direction
NEURAL_NETWORK_IMAGE_SIZE = 96

# camera 1 is the intel realsense, camera 2 is the black camera


class PingPongPipeline:

    def __init__(self):
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
        self.realsense.set(cv2.CAP_PROP_FRAME_HEIGHT, REALSENSE_IMAGE_DIM[0])

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
        image_points_realsense = []
        image_points_black_camera = []
        times = []
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
                if i < 30:
                    continue

                if prev_gray_img_realsense is not None:

                    ball_detected, point_realsense = self.blobDetector.find_ball(color_image_realsense)
                    ball_detected_2, point_black = self.blobDetector.find_ball(color_image_black_camera)

                    if ball_detected:
                        # orig_realsense = np.copy(color_image_realsense)
                        annotated_image_realsense = cv2.circle(color_image_realsense,
                                                              (int(point_realsense[1]), int(point_realsense[0])),
                                                              20,
                                                              (255, 0, 0),
                                                              2)
                        cv2.imshow("annotated_image_realsense", annotated_image_realsense)
                        cv2.imwrite(trajectory_path + "realsense" + str(i) + ".png", annotated_image_realsense)
                    else:
                        cv2.imshow("annotated_image_realsense", color_image_realsense)
                        # cv2.imwrite(trajectory_path + "realsense" + str(i) + ".png", color_image_realsense)

                    if ball_detected_2:
                        # orig_black = np.copy(color_image_black_camera)
                        annotated_image_black_camera = cv2.circle(color_image_black_camera,
                                                                  (int(point_black[1]), int(point_black[0])),
                                                                  20,
                                                                  (255, 0, 0),
                                                                  2)
                        cv2.imshow("annotated_image_black_camera", annotated_image_black_camera)
                        cv2.imwrite(trajectory_path + "black" + str(i) + ".png", annotated_image_black_camera)
                    else:
                        cv2.imshow("annotated_image_black_camera", color_image_black_camera)
                        # cv2.imwrite(trajectory_path + "black" + str(i) + ".png", color_image_black_camera)


                    cv2.waitKey(1)

                    if ball_detected and ball_detected_2:

                        if self.prev_point_black is not None and self.prev_point_realsense is not None:

                            ball_horizontal_speed = point_realsense[1] - self.prev_point_realsense[1]
                            print(ball_horizontal_speed)
                            # if ball has started to move towards the robot
                            if ball_horizontal_speed > MIN_VEL and trajectory_frame_count < TRAJECTORY_N_FRAMES:

                                image_points_realsense.append(point_realsense)
                                image_points_black_camera.append(point_black)

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
                                print(closest_pt)

                                if closest_pt is not None:
                                    prev_time = curr_time
                                    curr_time = time.perf_counter()

                                    if closest_pt[0] < 0.2:
                                        print("was not able to calculate trajectory in time")
                                        cannot_reach_ball = True
                                        break

                                    ball_dest_estimates.append([closest_pt[0], closest_pt[1], closest_pt[2]])
                                    times.append(curr_time)

                                    # if prev_closest_pt is not None:
                                        # ball_dest_at_x = \
                                        #     self.trajectoryCalculator.calculate_trajectory(
                                        #         np.array([closest_pt[0], closest_pt[1], closest_pt[2]]),
                                        #         np.array([prev_closest_pt[0], prev_closest_pt[1], prev_closest_pt[2]]),
                                        #         curr_time - prev_time,
                                        #         0.1)
                                        # if ball_dest_at_x is not None:
                                        #     ball_dest_estimates.append(ball_dest_at_x)
                                        #     trajectory_frame_count += 1
                                        # else:
                                        #     cannot_reach_ball = True
                                        #     break


                                    # prev_closest_pt = closest_pt

                                trajectory_frame_count += 1

                                if trajectory_frame_count == TRAJECTORY_N_FRAMES:
                                    break

                        self.prev_point_realsense = point_realsense
                        self.prev_point_black = point_black

                prev_gray_img_realsense = gray_image_realsense
                prev_gray_img_black_camera = gray_image_black_camera

            if cannot_reach_ball is False:
                # after collecting all the data, find the average estimate of the trajectory of the ball
                # ball_dest_estimates = np.array(ball_dest_estimates)
                # avg_dest = np.zeros((3, ))
                #
                # for i in range(3):
                #     avg_dest[i] = np.sum(ball_dest_estimates[:, i])/TRAJECTORY_N_FRAMES

                t = np.array(times)

                # print(ball_dest_estimates)
                ball_dest_estimates_np = np.array(ball_dest_estimates)

                fitx = np.polyfit(t, ball_dest_estimates_np[:, 0], 2)
                fity = np.polyfit(t, ball_dest_estimates_np[:, 1], 2)
                fitz = np.polyfit(t, ball_dest_estimates_np[:, 2], 2)

                x = 0.1
                time_at_x_arr = (np.roots(np.array([fitx[0], fitx[1], fitx[2] - x])))
                for arr_time in time_at_x_arr:
                    if t[2] < arr_time < t[2] + 0.7:
                        time_at_x = arr_time

                if np.imag(time_at_x) != 0:
                    time_at_x = np.real(time_at_x)

                y_at_x = fity[0] * time_at_x ** 2 + fity[1] * time_at_x + fity[2]
                z_at_x = fitz[0] * time_at_x ** 2 + fitz[1] * time_at_x + fitz[2]

                print(x, y_at_x, z_at_x, time_at_x, t, fitz)

                target_not_reachable, angles = self.invKin.analytical_inverse_kinematics(x, y_at_x, z_at_x, -0.7)
            else:
                target_not_reachable = True

            if target_not_reachable:
                print("target not reachable")
            else:
                self.publisher.publish_angles(angles)
                print("target reachable")

        finally:
            print("done")

if __name__ == "__main__":
    pipeline = PingPongPipeline()
    pipeline.go_to_ball()