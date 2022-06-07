import numpy as np
import cv2
from inverseKinematics import InverseKinematics
from colorChecker import ColorChecker
from ballDetector import BallDetector
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
TRAJECTORY_N_FRAMES = 1
MIN_VEL = 10 # in pixels/frame in the horizontal direction
NEURAL_NETWORK_IMAGE_SIZE = 96

# robot is on the right, player is on the left


count = 0

# black camera intrinsic parameters:

black_camera_matrix = np.array([
    [575.454025, 0.00000000e+00, 320],
    [0.00000000e+00, 574.53046, 240],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

black_camera_distortion = np.array([[ 0.06571678, -0.06531794, -0.00267922, -0.00469088, -0.05419306]])

# intel realsense intrinsic parameters:

realsense_camera_matrix = np.array([
    [591.48522865, 0., 320],
    [0., 591.9638662, 240],
    [0., 0., 1.]])

realsense_distortion = np.array([[ 0.00252699,  0.50539056,  0.00564689,  0.00742319, -1.62753842]])

# camera 1 is the intel realsense, camera 2 is the black camera

class PingPongPipeline:

    def __init__(self):
        # print("starting init function")
        self.invKin = InverseKinematics()
        self.ballDetector = BallDetector()
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

    def go_to_ball(self):
        global count

        # previous information used for motion tracking
        previous_ball_precise_location = (0, 0)  # in pixels
        previous_ball_precise_location_world_frame = (0, 0, 0)  # (x, y, z) in meters
        prev_gray_img_realsense = None
        prev_gray_img_black_camera = None
        prev_time = None
        curr_time = None
        prev_bbox_center_1 = None
        prev_bbox_center_2 = None
        prev_closest_pt = None
        closest_pt = None

        trajectory_frame_count = 0
        ball_dest_estimates = []
        dest_x_avg = None
        dest_y_avg = None
        dest_z_avg = None
        done = False
        final_ball_dest_estimate = (0, 0, 0)  # (x, y, z) in meters
        cannot_reach_ball = False

        i = 0

        # Streaming loop
        try:
            while True:
                # print(str(i))

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
                    ball_detected, xBox, yBox, wBox, hBox = self.find_ball(prev_gray_img_realsense,
                                                                           gray_image_realsense,
                                                                           color_image_realsense)

                    ball_detected_2, xBox2, yBox2, wBox2, hBox2 = self.find_ball(prev_gray_img_black_camera,
                                                                                gray_image_black_camera,
                                                                                color_image_black_camera)

                    realsense_valid = False
                    black_valid = False

                    if ball_detected:
                        # print("ball detected in camera 1")

                        # check if detection is valid
                        # print(color_image_realsense.shape)
                        # print(xBox, yBox, wBox, hBox)
                        realsense_valid = self.colorChecker.is_valid(
                            color_image_realsense[yBox:(yBox + hBox), xBox:(xBox + wBox)])

                        bbox_image_realsense = cv2.rectangle(color_image_realsense, (xBox, yBox), (xBox + wBox, yBox + hBox),
                                                   (255, 0, 0), 2)
                        cv2.imshow("bbox image realsense", bbox_image_realsense)
                    else:
                        cv2.imshow("bbox image realsense", color_image_realsense)

                    if ball_detected_2:
                        # print("ball detected in camera 2")

                        # check if detection is valid
                        black_valid = self.colorChecker.is_valid(
                            color_image_black_camera[yBox2:(yBox2 + hBox2), xBox2:(xBox2 + wBox2)])

                        bbox_image_black_camera = cv2.rectangle(color_image_black_camera, (xBox2, yBox2), (xBox2 + wBox2, yBox2 + hBox2),
                                                  (255, 0, 0), 2)
                        cv2.imshow("bbox image black camera", bbox_image_black_camera)
                    else:
                        cv2.imshow("bbox image black camera", color_image_black_camera)

                    cv2.waitKey(1)

                    # todo: find actual location of ball in blurred image based on previous pixel position
                    # todo: use actual time instead of computation time
                    # https: // stackoverflow.com / questions / 1557571 / how - do - i - get - time - of - a - python - programs - execution
                    # todo: make processing faster in order to make frames closer together

                    if realsense_valid and black_valid:
                        # find coordinates of center of ball in both cameras
                        r1 = int(yBox + hBox/2)
                        c1 = int(xBox + wBox/2)
                        bbox_center_1 = (r1, c1)

                        r2 = int(yBox2 + hBox2/2)
                        c2 = int(xBox2 + wBox2/2)
                        bbox_center_2 = (r2, c2)

                        cv2.imwrite(trajectory_path + "realsense_" + str(count) + ".png",
                                    bbox_image_realsense)
                        cv2.imwrite(trajectory_path + "black_" + str(count) + ".png",
                                    bbox_image_black_camera)

                        count += 1

                        if prev_bbox_center_1 is not None:
                            # velocity in units of pixels per frame
                            ball_horizontal_speed = (bbox_center_1[1] - prev_bbox_center_1[1])
                            # print(ball_horizontal_speed)
                            if ball_horizontal_speed > MIN_VEL and trajectory_frame_count < TRAJECTORY_N_FRAMES:

                                # position of ball in realsense camera frame, as estimated with realsense
                                x1_d, y1_d, z1_d = self.frameConverter.image_to_camera_frame("realsense", r1, c1)

                                # position of ball in black camera frame, as estimated with black camera
                                x2_e, y2_e, z2_e = self.frameConverter.image_to_camera_frame("black", r2, c2)

                                # convert both positions to robot arm frame:
                                x1_c, y1_c, z1_c = self.frameConverter.camera_to_robot_frame("realsense", x1_d, y1_d, z1_d)
                                x2_c, y2_c, z2_c = self.frameConverter.camera_to_robot_frame("black", x2_e, y2_e, z2_e)

                                closest_pt = self.frameConverter.find_intersection_point(x1_c, y1_c, z1_c, x2_c, y2_c, z2_c)

                                if closest_pt is not None:
                                    # print(closest_pt[0], closest_pt[1], closest_pt[2])
                                    # print("time diff: " + str(curr_time - prev_time))
                                    prev_time = curr_time
                                    curr_time = time.time()

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
                                                0.15)
                                        if ball_dest_at_x is not None:
                                            ball_dest_estimates.append(ball_dest_at_x)
                                            trajectory_frame_count += 1
                                        else:
                                            cannot_reach_ball = True
                                            break

                                    prev_closest_pt = closest_pt

                                if trajectory_frame_count == TRAJECTORY_N_FRAMES:
                                    break

                        prev_bbox_center_1 = bbox_center_1
                        prev_bbox_center_2 = bbox_center_2

                prev_gray_img_realsense = gray_image_realsense
                prev_gray_img_black_camera = gray_image_black_camera

            if cannot_reach_ball is False:
                # after collecting all the data, find the average estimate of the trajectory of the ball
                ball_dest_estimates = np.array(ball_dest_estimates)
                # print(ball_dest_estimates)
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

    def find_ball(self, prev_img_gray, curr_img_gray, curr_img_color):

        img_height = curr_img_color.shape[0]
        img_width = curr_img_color.shape[1]

        # use difference between frames for motion tracking
        frameDelta = cv2.absdiff(prev_img_gray, curr_img_gray)
        motion_img = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

        # remove salt and pepper noise
        motion_img = cv2.medianBlur(motion_img, 17)

        # masked_image = np.zeros(curr_img_color.shape, dtype=np.uint8)

        contours, hierarchy = cv2.findContours(motion_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_confidence = 0
        xmax = None
        ymax = None
        wmax = None
        hmax = None

        best_left_x = best_right_x = best_up_y = best_down_y = 0
        # print("number of contours: " + str(len(contours)))
        for contour in contours:
            # print("contour detected")
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

            if ball_detected:
                if confidence > max_confidence:
                    xmax = xBox
                    ymax = yBox
                    wmax = wBox
                    hmax = hBox
                    max_confidence = confidence
                    # best_left_x = left_x
                    # best_right_x = right_x
                    # best_up_y = up_y
                    # best_down_y = down_y


        # if max_confidence > 0:
        #     masked_image[best_up_y:best_down_y, best_left_x:best_right_x] = curr_img_color[best_up_y:best_down_y, best_left_x:best_right_x]
        #     bbox_image = cv2.rectangle(masked_image, (xmax, ymax), (xmax + wmax, ymax + hmax), (255, 0, 0), 2)
        #     # print(xmax, wmax, ymax, hmax, max_confidence)
        # #     cv2.imshow("masked motion img", masked_image)
        # #     cv2.waitKey(1000)
        # # else:
        #     # cv2.imshow("masked motion img", masked_image)
        #     # cv2.waitKey(1)

        if xmax is None:  # if no bounding boxes were detected
            return False, None, None, None, None
        else:
            return True, xmax, ymax, wmax, hmax

if __name__ == "__main__":
    pipeline = PingPongPipeline()
    pipeline.go_to_ball()