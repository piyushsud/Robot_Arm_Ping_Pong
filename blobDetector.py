import numpy as np
import cv2

class BlobDetector:

    def __init__(self):
        self.lower_orange = np.array([6, 70, 130])
        self.upper_orange = np.array([18, 255, 255])

        self.detector = cv2.SimpleBlobDetector()

        params = cv2.SimpleBlobDetector_Params()

        params.filterByArea = True
        params.minArea = 300
        params.maxArea = 2000
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.filterByColor = False

        self.detector = cv2.SimpleBlobDetector_create(params)

    def find_ball(self, img):
        image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        masked_image = cv2.inRange(image_hsv, self.lower_orange, self.upper_orange)

        original_image_masked = cv2.bitwise_and(img, img, mask=masked_image)
        original_image_masked = cv2.erode(original_image_masked, None, iterations=2)
        original_image_masked = cv2.dilate(original_image_masked, None, iterations=2)
        original_image_masked = cv2.dilate(original_image_masked, None, iterations=2)
        original_image_masked = cv2.erode(original_image_masked, None, iterations=2)
        grayscale_masked_image = cv2.cvtColor(original_image_masked, cv2.COLOR_BGR2GRAY)

        # Detect blobs.
        keypoints = self.detector.detect(grayscale_masked_image)

        if len(keypoints) > 0:
            return keypoints[0]
        else:
            return None
