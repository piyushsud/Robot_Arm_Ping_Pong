import cv2
import numpy as np
import time
import pyrealsense2 as rs

class BallDetector:

    def __init__(self):
        self.net = cv2.dnn.readNet(
            "C:/Users/piyus/Robot_Arm_Ping_Pong/darknet-master/build/darknet/x64/backup/yolov4-tiny-ping-pong_final.weights",
            "C:/Users/piyus/Robot_Arm_Ping_Pong/darknet-master/build/darknet/x64/cfg/yolov4-tiny-ping-pong.cfg")
        self.classes = []
        with open("C:/Users/piyus/Robot_Arm_Ping_Pong/darknet-master/build/darknet/x64/data/ping_pong.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        layer_names = self.net.getLayerNames()
        self.outputlayers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.frame_id = 0

    def find_ball_bbox(self, frame, depth_image, x_top_left, y_top_left):

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
                if confidence > 0.3:
                    # onject detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                    # rectangle co-ordinaters
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])  # put all rectangle areas
                    confidences.append(
                        float(confidence))  # how confidence was that object detected and show that percentage
                    class_ids.append(class_id)  # name of the object tha was detected

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)

        xmax = None
        ymax = None
        wmax = None
        hmax = None
        max_confidence = 0

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                center_x_box = int(x + w/2)
                center_y_box = int(y + h/2)
                depth = depth_image[center_y_box, center_x_box]/1000  # depth in meters
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # frame = cv2.resize(frame, (288, 288))
                # cv2.imshow("frame", frame)
                # cv2.waitKey(0)
                if confidence > max_confidence and 0.61 < depth < 1.67:
                    xmax = x
                    ymax = y
                    wmax = w
                    hmax = h

        if xmax is None:  # if no bounding boxes were detected
            return False, None, None, None, None
        else:
            #todo: check if bounding box is correct in ping_pong_main now that we passed on the correct values to this function

            # convert cropped image bounding box coordinates to whole image bounding box coordinates
            img_x = x_top_left + xmax
            img_y = y_top_left + ymax
            print(x_top_left, y_top_left)
            print(xmax, ymax)
            return True, img_x, img_y, wmax, hmax

if __name__ == '__main__':
    # try:
    #     app.run(main)
    # except SystemExit:
    #     pass
    ballDetector = BallDetector()
    image = cv2.imread("C:/Users/piyus/GrabCAD/Robot_Arm/yolo_formatted_data/valid_image_folder/img0.jpg")
    ballDetector.find_ball_bbox(image)