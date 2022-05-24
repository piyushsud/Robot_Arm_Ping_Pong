# all dimensions are in meters, angles are in radians
import modern_robotics as mr
import numpy as np

PI = 3.1415926535

l1 = 0.00475
l2 = 0.1528
l3 = 0.0225
l4 = 0.1422
l5 = 0.0415
l7 = 0.1483
l6 = 0.061087  # horizontal length of end effector
l8 = 0.0404  # vertical length of end effector

ZERO_OFFSET_JOINT1 = 79.8
ZERO_OFFSET_JOINT2 = 150
ZERO_OFFSET_JOINT3 = 14
ZERO_OFFSET_JOINT4 = 110
ZERO_OFFSET_JOINT5 = 190  # 123 is horizontal

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

    # yaw is the angle from the positive x-axis
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

        thetas = [theta1, theta2, theta3, theta4, theta5]

        T = np.round(mr.FKinSpace(M, np.transpose(S), thetas), 4)

        x_calc = T[0, 3]
        y_calc = T[1, 3]
        z_calc = T[2, 3]

        target_reachable = True

        if x_calc < 0.12 or z_calc < 0.1:
            target_reachable = False

        joint_1_angle = -(thetas[0] * 180 / PI) * (30 / 60) + ZERO_OFFSET_JOINT1
        joint_2_angle = (thetas[1] * 180 / PI) * (87 / 40) + ZERO_OFFSET_JOINT2
        joint_3_angle = (thetas[2] * 180 / PI) * (73 / 60) + ZERO_OFFSET_JOINT3
        joint_4_angle = -(thetas[3] * 180 / PI) + ZERO_OFFSET_JOINT4
        joint_5_angle = (thetas[4] * 180 / PI) + ZERO_OFFSET_JOINT5

        if joint_5_angle < 0 or joint_5_angle > 180:
            target_reachable = False

        angles = np.array([joint_1_angle, joint_2_angle, joint_3_angle,joint_4_angle, joint_5_angle])

        return target_reachable, angles

if __name__ == "__main__":
    ikin = InverseKinematics()
    # thetas_zero = ikin.analytical_inverse_kinematics(0.3561, -0.0238, 0.1887, 0) #zero position
    xtarget = 0.3561
    ytarget = -0.0238
    ztarget = 0.1887
    desired_yaw = 0
    thetas = np.round(ikin.analytical_inverse_kinematics(xtarget, ytarget, ztarget, desired_yaw), 3)
    print(thetas)

