import numpy as np
import cv2
from inverseKinematics import InverseKinematics
from blobDetector import BlobDetector
from ballDetector import BallDetector
from frameConverter import FrameConverter
from publisher import MqttPublisher
import pyrealsense2 as rs
import time
from trajectoryCalculator import TrajectoryCalculator
import matplotlib.pyplot as plt
import time

test_path = "C:/Users/piyus/Robot_Arm_Ping_Pong/misc/test_pics/"

BLACK_CAMERA_IMAGE_DIM = (480, 640)
REALSENSE_IMAGE_DIM = (480, 640)

FPS = 30
N_CHANNELS = 3
MOTION_THRESHOLD = 100
MAX_INTENSITY = 255
BALL_DEST_X_POSITION = 2 # in meters
TRAJECTORY_N_FRAMES = 4
MIN_VEL = 10 # in pixels/frame in the horizontal direction
NEURAL_NETWORK_IMAGE_SIZE = 96

# robot is on the right, player is on the left

# camera 1 is the intel realsense, camera 2 is the black camera

class PingPongPipeline:

    def __init__(self):
        # print("starting init function")
        self.invKin = InverseKinematics()
        self.ballDetector = BallDetector()
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
        # previous information used for motion tracking
        previous_ball_precise_location = (0, 0)  # in pixels
        previous_ball_precise_location_world_frame = (0, 0, 0)  # (x, y, z) in meters
        prev_gray_img_realsense = None
        prev_gray_img_black_camera = None
        prev_time = None
        prev_bbox_center = None
        prev_x_world = None
        prev_y_world = None
        prev_z_world = None

        trajectory_frame_count = 0
        ball_dest_estimates = []
        dest_x_avg = None
        dest_y_avg = None
        dest_z_avg = None
        done = False
        final_ball_dest_estimate = (0, 0, 0)  # (x, y, z) in meters

        i = 0

        # Streaming loop
        try:
            while True:
                # print(str(i))

                # get black camera frame
                ret_black, black_camera_frame = self.black_cam.read()

                # get realsense frame
                ret_realsense, color_image_realsense = self.realsense.read()

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

                    if ball_detected:
                        print("ball detected in camera 1")
                        bbox_image_realsense = cv2.rectangle(color_image_realsense, (xBox, yBox), (xBox + wBox, yBox + hBox),
                                                   (255, 0, 0), 2)
                        cv2.imshow("bbox image realsense", bbox_image_realsense)
                    else:
                        cv2.imshow("bbox image realsense", color_image_realsense)

                    if ball_detected_2:
                        print("ball detected in camera 2")
                        bbox_image_black_camera = cv2.rectangle(color_image_black_camera, (xBox2, yBox2), (xBox2 + wBox2, yBox2 + hBox2),
                                                  (255, 0, 0), 2)
                        cv2.imshow("bbox image black camera", bbox_image_black_camera)
                    else:
                        cv2.imshow("bbox image black camera", color_image_black_camera)

                    cv2.waitKey(1)

                    if ball_detected and ball_detected_2:
                            # find coordinates of center of ball in both cameras
                            r1 = int(yBox + hBox/2)
                            c1 = int(xBox + wBox/2)
                            bbox_center_1 = (r1, c1)

                            r2 = int(yBox2 + hBox2/2)
                            c2 = int(xBox2 + wBox2/2)
                            bbox_center_2 = (r2, c2)

                            # print(bbox_center)

                            if prev_bbox_center is not None:
                                # velocity in units of pixels per frame
                                ball_horizontal_velocity = bbox_center_1[1] - prev_bbox_center1[1]
                                if ball_horizontal_velocity > MIN_VEL and trajectory_frame_count < TRAJECTORY_N_FRAMES:

                                    # position of ball in realsense camera frame, as estimated with realsense
                                    x1_d, y1_d, z1_d = self.frameConverter.image_to_camera_frame("realsense", r1, c1)

                                    # position of ball in black camera frame, as estimated with black camera
                                    x2_e, y2_e, z2_e = self.frameConverter.image_to_camera_frame("black", r2, c2)

                                    # convert both positions to robot arm frame:
                                    x1_c, y1_c, z1_c = self.frameConverter.camera_to_robot_frame("realsense", x1_d, y1_d, z1_d)
                                    x2_c, y2_c, z2_c = self.frameConverter.camera_to_robot_frame("black", x2_e, y2_e, z2_e)

                                    x_c, y_c, z_c = self.frameConverter.find_intersection_point(x1_c, y1_c, z1_c, x2_c, y2_c, z2_c)

                                    x_world, y_world, z_world = self.frameConverter.camera_to_world_frame(x, y, z)

                                    curr_time = time.time()
                                    ball_dest_at_x = \
                                        self.trajectoryCalculator.calculate_trajectory(x_world, y_world, z_world,
                                                                                       prev_x_world, prev_y_world,
                                                                                       prev_z_world, curr_time - prev_time)
                                    ball_dest_estimates.append(ball_dest_at_x)
                                    trajectory_frame_count += 1
                                    prev_time = curr_time
                                if trajectory_frame_count == TRAJECTORY_N_FRAMES:
                                    x_sum = 0
                                    y_sum = 0
                                    z_sum = 0
                            #
                            #         for estimate in ball_dest_estimates:
                            #             x_sum += estimate[0]
                            #             y_sum += estimate[1]
                            #             z_sum += estimate[2]
                            #
                            #         dest_x_avg = x_sum / TRAJECTORY_N_FRAMES
                            #         dest_y_avg = y_sum / TRAJECTORY_N_FRAMES
                            #         dest_z_avg = z_sum / TRAJECTORY_N_FRAMES
                            #
                            #         done = True
                            #         break
                            # prev_bbox_center = bbox_center
                            # break

                    if done is True:
                        break

                prev_gray_img_realsense = gray_image_realsense
                prev_gray_img_black_camera = gray_image_black_camera

            x_robot, y_robot, z_robot = self.frameConverter.world_to_robot_frame(dest_x_avg, dest_y_avg, dest_z_avg)
            target_reachable, angles = self.invKin.analytical_inverse_kinematics(x_robot, y_robot, z_robot, 0)
            print(x_robot, y_robot, z_robot)
            if target_reachable:
                # self.publisher.publish_angles(angles)
                print("target reachable")
            else:
                print("target not reachable")

        finally:
            print("done")
            # self.pipeline.stop()

    def find_ball(self, prev_img_gray, curr_img_gray, curr_img_color):

        # print("finding ball")

        img_height = curr_img_color.shape[0]
        img_width = curr_img_color.shape[1]
        # print("finding ball")
        #
        # prev_img_gray = cv2.GaussianBlur(prev_img_gray, (11, 11), 0)
        # curr_img_gray = cv2.GaussianBlur(curr_img_gray, (11, 11), 0)

        # use difference between frames for motion tracking
        frameDelta = cv2.absdiff(prev_img_gray, curr_img_gray)
        motion_img = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

        # remove salt and pepper noise
        motion_img = cv2.medianBlur(motion_img, 17)

        # print(curr_img_color.dtype)

        masked_image = np.zeros(curr_img_color.shape, dtype=np.uint8)

        contours, hierarchy = cv2.findContours(motion_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # _, _, _, _, _, _ = self.ballDetector.find_ball_bbox(curr_img_color, 0, 0)

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