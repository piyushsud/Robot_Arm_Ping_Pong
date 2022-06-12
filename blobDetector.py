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
        # self.lower_red = np.array([])
        # 176, 227, 191

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
        masked_image = cv2.erode(masked_image, None, iterations=2)
        masked_image = cv2.dilate(masked_image, None, iterations=3)

        original_image_masked = cv2.bitwise_and(img, img, mask=masked_image)

        grayscale_masked_image = cv2.cvtColor(original_image_masked, cv2.COLOR_BGR2GRAY)


        # masked_image_racket = cv2.inRange(image_hsv, self.lower_red, self.upper_red)

        # plt.imshow(grayscale_masked_image)
        # plt.show()

        # Detect blobs.
        keypoints = self.detector.detect(grayscale_masked_image)

        if len(keypoints) == 0:
            # print("0 keypoints")
            return False, None
        elif len(keypoints) == 1:
            # print("1 keypoint")
            contours, hierarchy = cv2.findContours(masked_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # # find contour with brighter orange color
            # center = None
            # max_value = 0
            # max_value_area = None
            # for ind, con in enumerate(contours):
            #     img_copy = np.copy(img)
            #     if len(con) < 5:
            #         continue
            #     area = cv2.contourArea(con)
            #     if area < 400:
            #         continue
            #
            #     print(ind)
            #     draw_copy = np.copy(img_copy)
            #     cv2.drawContours(draw_copy, contours, ind, (255, 0, 0))
            #     cv2.imshow("contours", draw_copy)
            #     cv2.waitKey(0)

            return True, (int(keypoints[0].pt[1]), int(keypoints[0].pt[0]))
        else:
            contours, hierarchy = cv2.findContours(masked_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # find contour with brighter orange color
            center = None
            max_value = 0
            for ind, con in enumerate(contours):
                img_copy = np.copy(img)
                if len(con) < 5:
                    continue
                area = cv2.contourArea(con)
                if area < 400:
                    continue

                minEllipse = cv2.fitEllipse(con)
                x = int(minEllipse[0][0])
                y = int(minEllipse[0][1])

                left = x-60
                right = x+60
                up = y-60
                down = y+60

                if left < 0:
                    left = 0
                    right = 120
                if up < 0:
                    up = 0
                    down = 120
                if right > 640:
                    left = 640 - 120
                    right = 640
                if down > 480:
                    up = 480 - 120
                    down = 480

                cropped_img_before_ellipse = np.copy(img_copy[up:down, left:right])
                cv2.ellipse(img_copy, minEllipse, (0, 0, 0), -1)
                img_cropped = img_copy[up:down, left:right]
                gray_img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
                gray_img_cropped = 255 - gray_img_cropped
                ret, mask = cv2.threshold(gray_img_cropped, 254, 255, cv2.THRESH_BINARY)
                hsv_cropped = cv2.cvtColor(cropped_img_before_ellipse, cv2.COLOR_BGR2HSV)
                avg = np.nanmean(np.where(mask, hsv_cropped[:, :, 2], np.nan))
                if avg > max_value:

                    max_value = avg
                    center = (y, x)
            return True, center


if __name__ == "__main__":
    blobDetector = BlobDetector()
    for i in range(750, 950):
        img = cv2.imread("C:/Users/piyus/Robot_Arm_Ping_Pong/misc/trajectory_pics/black" + str(858) + ".png")
        found, pt = blobDetector.find_ball(img)
        if found:
            print(i)
            bbox_image = cv2.circle(img, pt, 30, (255, 0, 0), 5)
            cv2.imshow("image with bounding box", bbox_image)
            cv2.waitKey()
        else:
            print("not found")

