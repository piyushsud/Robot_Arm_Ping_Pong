import math
import mediapipe as mp
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

class BlobDetector:

    def __init__(self):
        self.lower_orange = np.array([4, 70, 120])
        self.upper_orange = np.array([14, 255, 255])

        params = cv2.SimpleBlobDetector_Params()

        params.filterByArea = True
        params.minArea = 400
        params.maxArea = 10000
        params.filterByCircularity = True
        params.minCircularity = 0.2
        params.filterByConvexity = True
        params.minConvexity = 0.7
        params.filterByInertia = True
        params.minInertiaRatio = 0.1
        params.filterByColor = False

        self.detector = cv2.SimpleBlobDetector_create(params)

    def find_ball(self, img):

        image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        masked_image = cv2.inRange(image_hsv, self.lower_orange, self.upper_orange)
        masked_image = cv2.dilate(masked_image, None, iterations=1)

        original_image_masked = cv2.bitwise_and(img, img, mask=masked_image)
        grayscale_masked_image = cv2.cvtColor(original_image_masked, cv2.COLOR_BGR2GRAY)

        # plt.imshow(grayscale_masked_image)
        # plt.show()

        # Detect blobs.
        keypoints = self.detector.detect(grayscale_masked_image)

        if len(keypoints) == 0:
            # print("0")
            return False, None
        elif len(keypoints) == 1:
            # print("1")
            return True, (int(keypoints[0].pt[0]), int(keypoints[0].pt[1]))
        else:
            # print("more than 1")
            contours, hierarchy = cv2.findContours(masked_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # find contour with brighter orange color
            center = None
            max_value = 0
            for con in contours:
                img_copy = np.copy(img)
                if len(con) < 5:
                    continue
                area = cv2.contourArea(con)
                if area < 400:
                    continue
                minEllipse = cv2.fitEllipse(con)
                x = int(minEllipse[0][0])
                y = int(minEllipse[0][1])

                cropped_img_before_ellipse = np.copy(img_copy[y-60:y+60, x-60:x+60])
                cv2.ellipse(img_copy, minEllipse, (0, 0, 0), -1)
                img_cropped = img_copy[y-60:y+60, x-60:x+60]
                gray_img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
                gray_img_cropped = 255 - gray_img_cropped
                ret, thresh = cv2.threshold(gray_img_cropped, 254, 255, cv2.THRESH_BINARY)
                avg = np.nanmean(np.where(thresh, cropped_img_before_ellipse[:, :, 2], np.nan))
                if avg > max_value:
                    max_value = avg
                    center = (x, y)

            return True, center


if __name__ == "__main__":
    blobDetector = BlobDetector()
    img = cv2.imread("C:/Users/piyus/Robot_Arm_Ping_Pong/misc/trajectory_pics/realsense194.png")
    found, pt = blobDetector.find_ball(img)
    if found:
        bbox_image = cv2.circle(img, pt, 30, (255, 0, 0), 5)
        cv2.imshow("image with bounding box", bbox_image)
        cv2.waitKey()
    else:
        print("not found")

