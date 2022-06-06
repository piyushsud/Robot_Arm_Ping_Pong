# all dimensions are in meters, angles are in radians
import modern_robotics as mr
import numpy as np
import pickle
# import board
# import busio
# import adafruit_pca9685
# from adafruit_servokit import ServoKit
import time

CALC_TOLERANCE = 0.1
STEP = 0.05
DIV = 100.0

# i2c = busio.I2C(board.SCL, board.SDA)
# hat = adafruit_pca9685.PCA9685(i2c)
# kit = ServoKit(channels=16)

ZERO_OFFSET_JOINT1 = 87
ZERO_OFFSET_JOINT2 = 172
ZERO_OFFSET_JOINT3 = 185
ZERO_OFFSET_JOINT4 = 135
ZERO_OFFSET_JOINT5 = 40

# kit.servo[11].set_pulse_width_range(800, 2200)  # joint 1
# kit.servo[10].set_pulse_width_range(900, 2100)  # joint 2
# kit.servo[13].set_pulse_width_range(553, 2520)  # joint 3
# kit.servo[14].set_pulse_width_range(553, 2425)  # joint 4
# kit.servo[12].set_pulse_width_range(553, 2425)  # joint 5
#
# kit.servo[11].actuation_range = 202  # joint 1, 0 is fully left, 202 is fully right
# kit.servo[10].actuation_range = 190  # joint 2, 0 is fully up, 190 is fully down (close to zero position)
# kit.servo[13].actuation_range = 190  # joint 3
# kit.servo[14].actuation_range = 190  # joint 4
# kit.servo[12].actuation_range = 190  # joint 5

actuation_ranges = [202, 190, 190, 190, 190]

PI = 3.1415926535

l1 = 0.0200
l2 = 0.1025
l3 = 0.0504
l4 = 0.0967
l5 = 0.0568
l7 = 0.1413
l6 = 0.0590  # horizontal length of end effector
l8 = 0.0422  # vertical length of end effector

with open('angles.pkl', 'rb') as file:
    currentPos = pickle.load(file)

TOLERANCE = 0.001

M = np.array([
    [1, 0, 0, l2 + l4 + l6],
    [0, 1, 0, -(l1 - l3 + l5)],
    [0, 0, 1, l7 + l8],
    [0, 0, 0, 1]
])

S = np.array([
    [0, 0, 1, 0, 0, 0],
    [0, 1, 0, -l7, 0, 0],
    [0, 1, 0, -l7, 0, l2],
    [0, 1, 0, -l7, 0, l2 + l4],
    [0, 0, 1, -(l5 - l3 + l1), -(l2 + l4), 0]
])


class InverseKinematics:

    def __init__(self):
        pass

    # yaw is the angle from the positive x axis

    def analytical_inverse_kinematics(self, x, y, z, yaw):
        xc = x - l6 * np.cos(yaw)
        yc = y - l6 * np.sin(yaw)
        zc = z

        # theta 1
        alpha = np.arctan2(yc, xc)
        d = l1 - l3 + l5
        beta = np.arcsin(d / np.sqrt(xc ** 2 + yc ** 2))
        theta1 = alpha + beta

        # theta 5
        theta5 = yaw - theta1

        # x3end, y3end, and z3end
        z3end = zc - l8

        frame_rot = np.array([
            [np.cos(theta1 + PI), -np.sin(theta1 + PI), 0],
            [np.sin(theta1 + PI), np.cos(theta1 + PI), 0],
            [0, 0, 1]])

        frame_translation = np.array([
            [1, 0, xc],
            [0, 1, yc],
            [0, 0, 1]])

        transform_mat = np.matmul(frame_translation, frame_rot)

        pt = np.array([0, -(l5 - l3 + l1), 1])

        pt_world_frame = np.matmul(transform_mat, pt)

        x3end = pt_world_frame[0]
        y3end = pt_world_frame[1]

        d2 = np.sqrt(x3end ** 2 + y3end ** 2 + (z3end - l7) ** 2)

        # theta 2
        gamma = np.arccos(np.clip((l4 ** 2 - l2 ** 2 - d2 ** 2) / (-2 * l2 * d2), -1, 1))
        theta2 = -(gamma + np.arcsin((z3end - l7) / d2))

        # theta 3
        phi = np.arccos(np.clip((d2 ** 2 - l4 ** 2 - l2 ** 2) / (-2 * l4 * l2), -1, 1))
        theta3 = PI - phi

        # theta 4
        theta4 = -(theta2 + theta3)

        target_not_reachable = False

        thetas = np.round(np.array([theta1, theta2, theta3, theta4, theta5]), 3)

        T = np.round(mr.FKinSpace(M, np.transpose(S), thetas), 4)

        x_reached = T[0, 3]
        y_reached = T[1, 3]
        z_reached = T[2, 3]

        if abs(x_reached - x) > TOLERANCE or abs(y_reached - y) > TOLERANCE or abs(z_reached - z) > TOLERANCE:
            target_not_reachable = True
            print("bounded by kinematics")

        joint_1_angle = -(theta1 * 180 / PI) + ZERO_OFFSET_JOINT1
        joint_2_angle = (theta2 * 180 / PI) + ZERO_OFFSET_JOINT2
        joint_3_angle = -(theta3 * 180 / PI) + ZERO_OFFSET_JOINT3
        joint_4_angle = (theta4 * 180 / PI) + ZERO_OFFSET_JOINT4
        joint_5_angle = -(theta5 * 180 / PI) + ZERO_OFFSET_JOINT5

        angles = np.array([joint_1_angle, joint_2_angle, joint_3_angle, joint_4_angle, joint_5_angle])
        # print(angles)
        servo_nums = [11, 10, 13, 14, 12]
        for i, angle in enumerate(angles):
            if angle < 0 or angle > actuation_ranges[i]:
                target_not_reachable = True

        return target_not_reachable, angles


    def move_to_target(self, angles, servo_nums):
        dist = np.subtract(angles, currentPos)
        for j in range(int(DIV)):
            for i, angle in enumerate(angles):
                currentPos[i] += dist[i] / DIV
                kit.servo[servo_nums[i]].angle = currentPos[i]
        with open('angles.pkl', 'wb') as file:
            pickle.dump(currentPos, file)


    def slow_move(self, angle, servo_num, servo_nums):
        i = 0
        while True:
            if servo_nums[i] == servo_num:
                break
            i += 1

        while True:
            dist = angle - currentPos[i]
            if abs(dist) > CALC_TOLERANCE:
                sign = abs(dist) / dist
                currentPos[i] += STEP * sign
                kit.servo[servo_nums[i]].angle = currentPos[i]
                # print(dist)
            else:
                print("reached")
                with open('angles.pkl', 'wb') as file:
                    pickle.dump(currentPos, file)
                break


if __name__ == "__main__":
    invKin = InverseKinematics()

    xtarget = 0.19
    ytarget = -0.1
    ztarget = 0.2
    desired_yaw = -0.7
    thetas = np.round(invKin.analytical_inverse_kinematics(xtarget, ytarget, ztarget, desired_yaw), 3)

    T = np.round(mr.FKinSpace(M, np.transpose(S), thetas), 4)

    x = T[0, 3]
    y = T[1, 3]
    z = T[2, 3]

    target_not_reachable = False

    if abs(x - xtarget) > TOLERANCE or abs(y - ytarget) > TOLERANCE or abs(z - ztarget) > TOLERANCE:
        target_not_reachable = True
        print("bounded by kinematics")

    joint_1_angle = -(thetas[0] * 180 / PI) + ZERO_OFFSET_JOINT1
    joint_2_angle = (thetas[1] * 180 / PI) + ZERO_OFFSET_JOINT2
    joint_3_angle = -(thetas[2] * 180 / PI) + ZERO_OFFSET_JOINT3
    joint_4_angle = (thetas[3] * 180 / PI) + ZERO_OFFSET_JOINT4
    joint_5_angle = -(thetas[4] * 180 / PI) + ZERO_OFFSET_JOINT5

    angles = np.array([joint_1_angle, joint_2_angle, joint_3_angle, joint_4_angle, joint_5_angle])
    print(angles)
    servo_nums = [11, 10, 13, 14, 12]
    for i, angle in enumerate(angles):
        if angle < 0 or angle > actuation_ranges[i]:
            target_not_reachable = True

    if target_not_reachable is False:
        print("moving towards target... ")
    else:
        print("target not reachable")





