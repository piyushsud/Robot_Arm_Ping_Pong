import cv2
import numpy as np
import matplotlib.pyplot as plt

NEURAL_NETWORK_IMAGE_SIZE = 96


class BlobDetector:

    def __init__(self):
        sensitivity = 90
        self.detector = cv2.SimpleBlobDetector()
        # self.lower_white = np.array([0, 0, 255 - sensitivity])
        # self.upper_white = np.array([255, sensitivity, 255])

        self.lower_orange = np.array([6, 70, 130])
        self.upper_orange = np.array([18, 255, 255])

        params = cv2.SimpleBlobDetector_Params()

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 500
        params.maxArea = 5000
        # Filter by Circularity
        params.filterByCircularity = False
        # params.minCircularity = 0.5
        # # Filter by Convexity
        params.filterByConvexity = False
        # params.minConvexity = 0.6
        # # # Filter by Inertia
        params.filterByInertia = False
        params.filterByColor = False
        # params.minInertiaRatio = 0.08
        # Create a detector with the parameters
        self.detector = cv2.SimpleBlobDetector_create(params)

    def find_ball_bbox(self, img, x_top_left, y_top_left):

        img = cv2.imread("C:/Users/piyus/Robot_Arm_Ping_Pong/misc/orange_ball_pics/im2.png")

        image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        masked_image = cv2.inRange(image_hsv, self.lower_orange, self.upper_orange)
        # masked_image = cv2.inRange(image_hsv, self.lower_white, self.upper_white)

        original_image_masked = cv2.bitwise_and(img, img, mask=masked_image)
        original_image_masked = cv2.erode(original_image_masked, None, iterations=2)
        original_image_masked = cv2.dilate(original_image_masked, None, iterations=2)
        original_image_masked = cv2.dilate(original_image_masked, None, iterations=2)
        original_image_masked = cv2.erode(original_image_masked, None, iterations=2)
        grayscale_masked_image = cv2.cvtColor(original_image_masked, cv2.COLOR_BGR2GRAY)

        # grayscale_masked_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # im_b = img[:, :, 0]
        # im_g = img[:, :, 1]
        # im_r = img[:, :, 2]
        # cv2.imshow("img", grayscale_img)
        # cv2.imshow("img blue", thresh)

        # Detect blobs.
        keypoints = self.detector.detect(grayscale_masked_image)
        # print(keypoints)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #
        plt.imshow(img)
        plt.show()
        cv2.imshow("keypoints", im_with_keypoints)
        cv2.imshow("masked image", original_image_masked)
        cv2.imshow("color image", img)
        # cv2.imwrite("C:/Users/piyus/Robot_Arm_Ping_Pong/misc/orange_ball_pics/im2.png", img)
        cv2.waitKey(1)

        return False, None, None, None, None, None


if __name__ == '__main__':
    ballDetector = BlobDetector()
    # image = cv2.imread("C:/Users/piyus/GrabCAD/Robot_Arm/yolo_formatted_data/valid_image_folder/img340.jpg")
    image = cv2.imread("C:/Users/piyus/Robot_Arm_Ping_Pong/misc/test_pics/8.jpg")
    ballDetector.find_ball_bbox(image, 0, 0)
    # bbox_image = cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 1)
    # cv2.imshow("image with bounding box", cv2.resize(bbox_image, (96*4, 96*4)))
    # cv2.waitKey()

