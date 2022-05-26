import cv2
import numpy as np

# checkerboard upright frame = frame A
# checkerboard on table frame = frame B
# robot arm frame = frame C
# realsense frame = frame D
# black camera frame = frame E

# in units of SQUARE_SIZE
T_ad = np.array([[0.99929197, 0.02608819, 0.02711012, -1.47726141],
                 [-0.02659131, 0.99947765, 0.01836679, -1.74899115],
                 [-0.0266168, -0.01907468,  0.99946371, 20.24589362],
                 [0, 0, 0, 1]])

T_be = np.array([[0.99965129, -0.01992817, -0.01732549, 3.1662789],
                 [0.01924965,  0.99907374, -0.03848515, -3.18508289],
                 [0.01807638,  0.03813822,  0.99910896, 23.29714278],
                 [0, 0, 0, 1]])

T_ca = np.array(
    [
        [-1, 0, 0, 0.4534875],
        [0, 0, -1, -0.2311625],
        [0, -1, 0, 0.2],
        [0, 0, 0, 1]
    ]
)

T_cb = np.array(
    [
        [1, 0, 0, 0.4525],
        [0, -1, 0, 0.0545],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ]
)


# T_ab * p_b changes the reference frame of p from b to a.
p_d = np.array([0, 0, 0, 1])
p_a = np.matmul(T_ad, p_d)
p_c1 = np.matmul(T_ca, p_a)

p_e = np.array([0, 0, 0, 1])
p_b = np.matmul(T_be, p_e)
p_c2 = np.matmul(T_cb, p_b)

print(p_c1)
print(p_c2)