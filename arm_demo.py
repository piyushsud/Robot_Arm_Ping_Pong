from inverseKinematics import InverseKinematics
from publisher import MqttPublisher
import numpy as np
import time

class Demo:

    def __init__(self):
        self.invKin = InverseKinematics()
        self.publisher = MqttPublisher()

    def run(self):
        time.sleep(5)
        ret, curr_angles = self.invKin.analytical_inverse_kinematics(0.2, 0, 0.18, -0.8)
        for i in range(25):
            reachable, angles = self.invKin.analytical_inverse_kinematics(0.2, 0.2, 0.18, -0.8)
            print(reachable, angles)
            new_angles = np.zeros((5, ), dtype=np.float64)
            for j in range(5):
                diff = angles[j] - curr_angles[j]
                new_angles[j] = curr_angles[j] + (i/100)*diff
            time.sleep(0.01)

            if reachable:
                self.publisher.publish_angles(new_angles)

        ret, curr_angles = self.invKin.analytical_inverse_kinematics(0.2, 0.2, 0.18, -0.8)
        for i in range(25):
            reachable, angles = self.invKin.analytical_inverse_kinematics(0.2, 0, 0.18, -0.8)
            print(reachable, angles)
            new_angles = np.zeros((5, ), dtype=np.float64)
            for j in range(5):
                diff = angles[j] - curr_angles[j]
                new_angles[j] = curr_angles[j] + (i/100)*diff
            time.sleep(0.01)

            if reachable:
                self.publisher.publish_angles(new_angles)
if __name__ == "__main__":
    demo = Demo()
    demo.run()