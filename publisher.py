#!/usr/bin/env python
# -*- coding: utf-8 -*-

import paho.mqtt.publish as publish
import numpy as np

class MqttPublisher:
    def __init__(self):
        self.host = "192.168.86.203"

    def publish_angles(self, angles):
        publish.single(topic="angles/data", payload=angles.tobytes(), hostname=self.host)

if __name__ == '__main__':
    host = "192.168.86.203"
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # publish a single message
    publish.single(topic="kids/yolo", payload=arr.tobytes(), hostname=host)

