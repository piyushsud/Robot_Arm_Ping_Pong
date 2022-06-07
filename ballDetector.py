# this NN is not robust to different lighting conditions, resolutions, and backgrounds.

import cv2
import numpy as np
import time
import pyrealsense2 as rs

CONFIDENCE_THRESH = 0.05
NMS_THRESH = 0.4

class BallDetector:

    def __init__(self):
        self.net = cv2.dnn.readNet(
            "C:/Users/piyus/Robot_Arm_Ping_Pong/darknet-master/build/darknet/x64/backup/yolov4-tiny-ping-pong_final.weights",
            "C:/Users/piyus/Robot_Arm_Ping_Pong/darknet-master/build/darknet/x64/cfg/yolov4-tiny-ping-pong.cfg")
        self.classes = []
        with open("C:/Users/piyus/Robot_Arm_Ping_Pong/darknet-master/build/darknet/x64/data/ping_pong.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        layer_names = self.net.getLayerNames()
        self.outputlayers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.frame_id = 0

    def find_ball_bbox(self, color_image, x_top_left, y_top_left):
        frame = color_image.copy()
        self.frame_id += 1

        height, width, channels = frame.shape

        # preprocess image
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (96, 96), (0, 0, 0), True, crop=False)

        # network forward pass
        self.net.setInput(blob)
        outs = self.net.forward(self.outputlayers)
        # print(outs[1])

        # Showing info on screen/ get confidence score of algorithm in detecting an object in blob
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > CONFIDENCE_THRESH:
                    # object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                    # rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    if x < 0:
                        w += x
                        x = 0

                    if x >= width:
                        diff = x - width - 1
                        w -= diff
                        x = width - 1

                    if y < 0:
                        h += y
                        y = 0

                    if y >= height:
                        diff = y - width - 1
                        h -= diff
                        y = height - 1

                    boxes.append([x, y, w, h])  # put all rectangle areas
                    confidences.append(
                        float(confidence))  # how confidence was that object detected and show that percentage
                    class_ids.append(class_id)  # name of the object tha was detected

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESH, NMS_THRESH)

        xmax = None
        ymax = None
        wmax = None
        hmax = None
        max_confidence = 0

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                confidence = confidences[i]
                if confidence > max_confidence:
                    xmax = x
                    ymax = y
                    wmax = w
                    hmax = h
                    max_confidence = confidence

        if xmax is None:  # if no bounding boxes were detected
            return False, None, None, None, None, None
        else:
            # convert cropped image bounding box coordinates to whole image bounding box coordinates

            img_x = x_top_left + xmax
            img_y = y_top_left + ymax
            # print(img_x, img_y, x_top_left, y_top_left, xmax, ymax, wmax, hmax)
            return True, img_x, img_y, wmax, hmax, max_confidence

if __name__ == '__main__':
    ballDetector = BallDetector()
    # image = cv2.imread("C:/Users/piyus/GrabCAD/Robot_Arm/yolo_formatted_data/valid_image_folder/img340.jpg")
    img = cv2.imread("C:/Users/piyus/Robot_Arm_Ping_Pong/misc/orange_ball_pics/im1.png")
    color_image = img[0:384, (640 - 384):640]
    found, x, y, w, h, conf = ballDetector.find_ball_bbox(color_image, 0, 0)
    bbox_image = cv2.rectangle(color_image, (x, y), (x+w, y+h), (255, 0, 0), 1)
    cv2.imshow("image", cv2.resize(color_image, (96*4, 96*4)))
    cv2.imshow("image with bounding box", cv2.resize(bbox_image, (96*4, 96*4)))
    cv2.waitKey()
