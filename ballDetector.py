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

        # loading image
        font = cv2.FONT_HERSHEY_PLAIN
        starting_time = time.time()
        self.frame_id = 0

    def find_ball_bbox(self, image, x, y):

        frame = image
        self.frame_id += 1

        height, width, channels = frame.shape
        # detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (96, 96), (0, 0, 0), True, crop=False)

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

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                color = self.colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 1, (255, 255, 255), 2)

        elapsed_time = time.time() - self.starting_time


        cv2.imshow("Image", frame)
        key = cv2.waitKey(1)  # wait 1ms the loop will start again and we will process the next frame

        if key == 27:  # esc key stops the process
            break;

if __name__ == '__main__':
    # try:
    #     app.run(main)
    # except SystemExit:
    #     pass
    ballDetector = BallDetector()
    image = cv2.imread("C:/Users/piyus/GrabCAD/Robot_Arm/yolo_formatted_data/valid_image_folder/img0.jpg")
    ballDetector.find_ball_bbox(image)